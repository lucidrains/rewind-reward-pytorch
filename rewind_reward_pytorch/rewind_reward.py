from __future__ import annotations
from itertools import chain

import torch
from torch import nn, tensor
from torch.nn import Module
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from x_transformers import Decoder, Encoder

from x_mlps_pytorch import Feedforwards

from sentence_transformers import SentenceTransformer

from transformers import AutoImageProcessor, AutoModel

from vit_pytorch.accept_video_wrapper import AcceptVideoWrapper

import einx
from einops import rearrange, pack, unpack

# helpers

def exists(v):
    return v is not None

def mask_from_lens(lens):
    seq = torch.arange(lens.amax().item(), device = lens.device)
    mask = einx.less('n, b -> b n', seq, lens)
    return mask

# their main proposal is just in Figure 9
# basically the gist is predict progress from video frames for dense rewards

class DinoImageEmbedder(Module):
    def __init__(
        self,
    ):
        super().__init__()

        self.image_processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
        self.image_model = AutoModel.from_pretrained('facebook/dinov2-base')

    def forward(self, images):
        model_inputs = self.image_processor(images, return_tensors = 'pt')

        outputs = self.image_model(**model_inputs)

        last_hidden_states = outputs[0]

        return last_hidden_states[:, 0] # cls

class RewardModel(Module):
    def __init__(
        self,
        encoder: dict | Encoder = dict(
            dim = 768,
            depth = 4,
            heads = 8,
            attn_dim_head = 64
        ),
        image_model: Module | None = None,
        mlp_predictor_depth = 3,
        reward_bins = 10,
        max_video_frames = 16,
        dim_image_embed = 768,
        lang_per_token_embed = True,
        sentence_transformer_path = 'sentence-transformers/all-MiniLM-L12-v2'
    ):
        super().__init__()

        self.lang_per_token_embed = lang_per_token_embed # whether to have token granularity

        self.mini_lm = SentenceTransformer(sentence_transformer_path)
        mini_lm_dim = self.mini_lm.encode(['__']).shape[-1]

        if not exists(image_model):
            image_model = DinoImageEmbedder()

        self.video_embed = AcceptVideoWrapper(
            image_model,
            add_time_pos_emb = True,
            time_seq_len = max_video_frames,
            dim_emb = dim_image_embed
        )

        self.encoder = Encoder(**encoder)
        dim = self.encoder.dim

        self.to_lang_tokens = nn.Linear(mini_lm_dim, dim)

        self.to_video_tokens = nn.Linear(dim_image_embed, dim)

        self.mlp_predictor = Feedforwards(
            dim = dim,
            dim_out = reward_bins,
            depth = mlp_predictor_depth
        )

    def parameters(self):
        return chain(
            self.encoder.parameters(),
            iter((self.video_embed.pos_emb,)),
            self.to_lang_tokens.parameters(),
            self.to_video_tokens.parameters(),
            self.mlp_predictor.parameters()
        )

    def forward(
        self,
        commands: list[str],
        video, # (b c t h w)
        rewards = None,
        video_lens = None
    ):
        assert len(commands) == video.shape[0]

        device = video.device
        mask = None

        # language embed

        lang_embeds = self.mini_lm.encode(
            commands,
            output_value = 'token_embeddings' if self.lang_per_token_embed else 'sentence_embedding',
            convert_to_numpy = False
        )

        lang_embeds = pad_sequence(lang_embeds, batch_first = True).to(device)

        if self.lang_per_token_embed:
            lens = tensor([t.shape[0] for t in lang_embeds], device = device)
            mask = mask_from_lens(lens)

        # video embeds

        video_embeds = self.video_embed(video, eval_with_no_grad = True)

        if self.lang_per_token_embed:
            mask = F.pad(mask, (0, video_embeds.shape[1]), value = True)

        # linear projections

        lang_tokens = self.to_lang_tokens(lang_embeds)

        video_tokens = self.to_video_tokens(video_embeds)

        tokens, lang_video_packed_shape = pack((lang_tokens, video_tokens), 'b * d')

        # attention

        attended = self.encoder(tokens, mask = mask)

        # unpack and project the video tokens to logits to train reward predictor

        _, attended_video_tokens = unpack(attended, lang_video_packed_shape, 'b * d')

        video_frame_logits = self.mlp_predictor(attended_video_tokens)

        # return raw prediction or loss
        # depending on whether `rewards` is passed in

        return_loss = exists(rewards)

        if not return_loss:
            return video_frame_logits

        # determine video masking for loss

        if exists(video_lens):
            video_mask = mask_from_lens(video_lens)

            max_video_len = video_lens.amax().item()
            video_frame_logits = video_frame_logits[:, :max_video_len]
            rewards = rewards[:, :max_video_len]

            rewards = einx.where('b t, b t,', video_mask, rewards, -1)

        # calculate loss

        loss = F.cross_entropy(
            rearrange(video_frame_logits, 'b t l -> b l t'),
            rewards,
            ignore_index = -1
        )

        return loss

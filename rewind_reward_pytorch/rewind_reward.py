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

from hl_gauss_pytorch import HLGaussLayer

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
        sentence_transformer_path = 'sentence-transformers/all-MiniLM-L12-v2',
        categorical_rewards = False,
        use_hl_gauss_loss = True,
        reward_min_value = 0.,
        reward_max_value = 5.,
        reward_hl_gauss_loss_num_bins = 20,
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
            dim_out = reward_bins if categorical_rewards else None,
            depth = mlp_predictor_depth
        )

        # whether to predict reward bins

        self.categorical_rewards = categorical_rewards

        # hl gauss loss or plain regression
        # https://arxiv.org/abs/2403.03950

        self.hl_gauss_layer = HLGaussLayer(
            dim = dim,
            use_regression = not use_hl_gauss_loss,
            hl_gauss_loss = dict(
                min_value = reward_min_value,
                max_value = reward_max_value,
                num_bins = reward_hl_gauss_loss_num_bins,
            )
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

        video_frame_embed_or_logits = self.mlp_predictor(attended_video_tokens)

        # determine video masking for loss

        if exists(video_lens):
            video_mask = mask_from_lens(video_lens)
            max_video_len = video_lens.amax().item()

            video_frame_embed_or_logits = video_frame_embed_or_logits[:, :max_video_len]

            if exists(rewards):
                rewards = rewards[:, :max_video_len]
                rewards = einx.where('b t, b t,', video_mask, rewards, -1)

        # naming

        if self.categorical_rewards:
            video_frame_logits = video_frame_embed_or_logits
        else:
            video_frame_embeds = video_frame_embed_or_logits

        # return raw prediction or loss
        # depending on whether `rewards` is passed in

        return_loss = exists(rewards)

        if not return_loss:
            if self.categorical_rewards:
                return video_frame_logits
            else:
                return self.hl_gauss_layer(video_frame_embeds)

        # calculate loss

        if self.categorical_rewards:
            assert rewards.dtype in (torch.long, torch.int)

            loss = F.cross_entropy(
                rearrange(video_frame_logits, 'b t l -> b l t'),
                rewards,
                ignore_index = -1
            )
        else:
            assert rewards.dtype == torch.float

            loss = self.hl_gauss_layer(video_frame_embeds, rewards, mask = video_mask)

        return loss

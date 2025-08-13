from __future__ import annotations
from itertools import chain
from random import random

import torch
from torch import nn, cat, tensor
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
from einops import rearrange, repeat, pack, unpack

# helpers

def exists(v):
    return v is not None

def satisfy_prob(prob):
    return random() < prob

def mask_from_lens(lens):
    seq = torch.arange(lens.amax().item(), device = lens.device)
    mask = einx.less('n, b -> b n', seq, lens)
    return mask

def randint(
    min_value: int,
    max_value: Tensor
):
    value_range = (max_value - min_value).float()
    return ((value_range * torch.rand_like(value_range)) + min_value).round().clamp(min = min_value).long()

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
        decoder: dict | Decoder = dict(
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
        num_register_tokens = 4, # register / memory tokens, can't hurt
        lang_per_token_embed = True,
        sentence_transformer_path = 'sentence-transformers/all-MiniLM-L12-v2',
        categorical_rewards = False,
        use_hl_gauss_loss = True,
        reward_min_value = 0.,
        reward_max_value = 1.,
        reward_hl_gauss_loss_num_bins = 20,
    ):
        super().__init__()

        self.lang_per_token_embed = lang_per_token_embed # whether to have token granularity

        self.mini_lm = SentenceTransformer(sentence_transformer_path)
        mini_lm_dim = self.mini_lm.encode(['__']).shape[-1]

        if not exists(image_model):
            image_model = DinoImageEmbedder()

        self.video_embed = AcceptVideoWrapper(image_model)


        self.decoder = Decoder(**decoder)
        dim = self.decoder.dim

        self.first_pos_emb = nn.Parameter(torch.randn(dim) * 1e-2) # only first frame gets a positional embed, so it cannot cheat on predicting progress

        self.to_lang_tokens = nn.Linear(mini_lm_dim, dim)

        self.to_video_tokens = nn.Linear(dim_image_embed, dim)

        self.mlp_predictor = Feedforwards(
            dim = dim,
            dim_out = reward_bins if categorical_rewards else None,
            depth = mlp_predictor_depth
        )

        # register / memory / attention sink tokens

        self.num_register_tokens = num_register_tokens
        self.register_tokens = nn.Parameter(torch.randn(num_register_tokens, dim) * 1e-2)

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
            self.decoder.parameters(),
            iter((self.video_embed.pos_emb,)),
            self.to_lang_tokens.parameters(),
            self.to_video_tokens.parameters(),
            self.mlp_predictor.parameters(),
            self.hl_gauss_layer.parameters()
        )

    def forward(
        self,
        commands: list[str],
        video,                        # (b c t h w)
        extra_embed_tokens = None,    # (b n d)
        rewards = None,
        video_lens = None
    ):
        batch = video.shape[0]
        assert len(commands) == batch

        device = video.device
        mask = None

        # register tokens

        register_tokens = repeat(self.register_tokens, 'n d -> b n d', b = batch)

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

        # extra embeds, whether they be RGBD (Adapt3R tokens) or Tactile tokens

        if not exists(extra_embed_tokens):
            extra_embed_tokens = register_tokens[:, 0:0] # easy way to create an empty (b 0 d) tensor

        elif exists(extra_embed_tokens) and exists(mask):
            mask = F.pad(mask, (0, extra_embed_tokens.shape[-2]), value = True)

        # video embeds

        video_embeds = self.video_embed(video, eval_with_no_grad = True)

        if self.lang_per_token_embed:
            mask = F.pad(mask, (0, video_embeds.shape[1] + self.num_register_tokens), value = True)

        # linear projections

        lang_tokens = self.to_lang_tokens(lang_embeds)

        video_tokens = self.to_video_tokens(video_embeds)

        # add video start positional embedding

        first_video_token, rest_video_tokens = video_tokens[:, :1], video_tokens[:, 1:]

        first_video_token = first_video_token + repeat(self.first_pos_emb, 'd -> b 1 d', b = batch)

        video_tokens = cat((first_video_token, rest_video_tokens), dim = 1)

        # pack all tokens for attention

        tokens, lang_video_packed_shape = pack((lang_tokens, register_tokens, extra_embed_tokens, video_tokens), 'b * d')

        # attention

        attended = self.decoder(tokens, mask = mask)

        # unpack and project the video tokens to logits to train reward predictor

        _, _, _, attended_video_tokens = unpack(attended, lang_video_packed_shape, 'b * d')

        video_frame_embed_or_logits = self.mlp_predictor(attended_video_tokens)

        # determine video masking for loss

        video_mask = None

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

# rewind wrapper

class RewindTrainWrapper(Module):
    def __init__(
        self,
        reward_model: RewardModel,
        rewind_augmentation_prob = 0.5
    ):
        super().__init__()
        assert not reward_model.categorical_rewards

        self.reward_model = reward_model

        self.rewind_augmentation_prob = rewind_augmentation_prob

    def forward(
        self,
        commands: list[str],
        video,                        # (b c t h w)
        extra_embed_tokens = None,    # (b n d)
        video_lens = None
    ):
        batch, max_video_len, device = video.shape[0], video.shape[2], video.device

        if not exists(video_lens):
            video_lens = torch.full((batch,), max_video_len, device = device).float()

        assert (video_lens >= 2).all(), 'you have an invalid single frame video in here'

        video_frame_seq = torch.arange(max_video_len, device = device)

        progress = einx.divide('n, b -> b n', video_frame_seq, video_lens - 1.)

        if satisfy_prob(self.rewind_augmentation_prob):

            rand_index = randint(2, video_lens)
            rewind_len = randint(1, rand_index - 1)

            video_lens = rand_index + rewind_len # the video lens are now up to that selected random index (i in paper) with the rewind length (k in paper)

            video_seq = torch.arange(max_video_len, device = device)
            reversed_video_seq = video_seq.flip(dims = (-1,))

            frame_indices = cat((video_seq, reversed_video_seq[1:]))

            aug_max_video_len = video_lens.amax().item()

            seq = torch.arange(aug_max_video_len, device = device)

            indices = einx.add('b, n -> b n', max_video_len - rand_index, seq)

            sel_indices_offsetted = frame_indices[indices]
            sel_frame_indices = einx.subtract('b n, b', sel_indices_offsetted, sel_indices_offsetted[:, 0])

            sel_frame_indices.clamp_(min = 0)

            batch_arange = torch.arange(batch, device = device)[:, None]

            video = rearrange(video, 'b c t ... -> b t c ...')
            video = video[batch_arange, sel_frame_indices]
            video = rearrange(video, 'b t c ... -> b c t ...')

            # i think it comes out to the same as just selecting out the forward progress from above, but please open an issue if not
            progress = progress[batch_arange, sel_frame_indices]

        loss = self.reward_model(
            commands = commands,
            video = video,
            video_lens = video_lens,
            rewards = progress
        )

        return loss

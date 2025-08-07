from __future__ import annotations

import torch
from torch import nn
from torch.nn import Module

from x_transformers import Decoder, Encoder

from x_mlps_pytorch import Feedforwards

from sentence_transformers import SentenceTransformer

from PIL import Image
from transformers import AutoImageProcessor, AutoModel

from vit_pytorch.accept_video_wrapper import AcceptVideoWrapper

from einops import pack, unpack

# helpers

def exists(v):
    return v is not None

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
        sentence_transformer_path = 'sentence-transformers/all-MiniLM-L12-v2'
    ):
        super().__init__()

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

    def forward(
        self,
        commands: list[str],
        video, # (b c t h w)
    ):
        device = video.device

        lang_embeds_npy = self.mini_lm.encode(commands)

        lang_embeds = torch.from_numpy(lang_embeds_npy).to(device)

        video_embeds = self.video_embed(video, eval_with_no_grad = True)

        # linear projections

        lang_tokens = self.to_lang_tokens(lang_embeds)

        video_tokens = self.to_video_tokens(video_embeds)

        tokens, lang_video_packed_shape = pack((lang_tokens, video_tokens), 'b * d')

        # attention

        attended = self.encoder(tokens)

        # unpack and project the video tokens to logits to train reward predictor

        _, attended_video_tokens = unpack(attended, lang_video_packed_shape, 'b * d')

        video_frame_logits = self.mlp_predictor(attended_video_tokens)

        return video_frame_logits

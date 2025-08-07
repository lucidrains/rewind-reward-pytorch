import torch
from torch.nn import Module

from x_transformers import Decoder, Encoder

from x_mlps_pytorch import FeedForwards

from sentence_transformers import SentenceTransformer

from PIL import Image
from transformers import AutoImageProcessor, AutoModel

from vit_pytorch.accept_video_wrapper import AcceptVideoWrapper

# their main proposal is just in Figure 9
# basically the gist is predict progress from video frames for dense rewards

class CrossModalSequentialAggregator(Module):
    def __init__(
        self,
        encoder: Encoder,
        mlp_predictor_depth = 3,
        reward_bins = 10,
        sentence_transformer_path = 'sentence-transformers/all-MiniLM-L12-v2'
    ):
        super().__init__()

        self.mini_lm = SentenceTransformer(sentence_transformer_path)

        self.image_processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
        self.image_model = AutoModel.from_pretrained('facebook/dinov2-base')

        self.mlp_predictor = FeedForwards(
            dim = encoder.dim,
            dim_out = reward_bins,
            depth = mlp_predictor_depth
        )

    def forward(
        self,
        commands: list[str]
    ):
        embeds = self.mini_lm.encode(commands)

import pytest
import torch

param = pytest.mark.parametrize

@param('token_embed', (False, True))
def test_rewind_reward(
    token_embed
):
    from rewind_reward_pytorch.rewind_reward import RewardModel

    reward_model = RewardModel(
        reward_bins = 10,
        lang_per_token_embed = token_embed
    )

    commands = [
      'pick up the blue ball and put it in the red tray',
      'pick up the red cube and put it in the green bin'
    ]

    video = torch.rand(2, 3, 16, 224, 224)

    logits = reward_model(commands, video) # (2, 16, 10)

    assert logits.shape == (2, 16, 10)

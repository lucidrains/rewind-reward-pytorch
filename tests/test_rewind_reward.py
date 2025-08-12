import pytest
import torch

param = pytest.mark.parametrize

@param('token_embed', (False, True))
def test_categorical_rewind_reward(
    token_embed
):
    from rewind_reward_pytorch.rewind_reward import RewardModel

    reward_model = RewardModel(
        reward_bins = 10,
        lang_per_token_embed = token_embed,
        categorical_rewards = True
    )

    commands = [
      'pick up the blue ball and put it in the red tray',
      'pick up the red cube and put it in the green bin'
    ]

    video = torch.rand(2, 3, 16, 224, 224)

    logits = reward_model(commands, video) # (2, 16, 10)

    assert logits.shape == (2, 16, 10)

@param('token_embed', (False, True))
@param('use_extra_embed_tokens', (False, True))
def test_rewind_reward(
    token_embed,
    use_extra_embed_tokens
):
    from rewind_reward_pytorch.rewind_reward import RewardModel

    reward_model = RewardModel(
        reward_bins = 10,
        lang_per_token_embed = token_embed,
        categorical_rewards = False
    )

    commands = [
      'pick up the blue ball and put it in the red tray',
      'pick up the red cube and put it in the green bin',
    ]

    video = torch.rand(2, 3, 16, 224, 224)

    extra_embed_tokens = None
    if use_extra_embed_tokens:
        extra_embed_tokens = torch.randn(2, 7, 768)

    loss = reward_model(
        commands,
        video,
        extra_embed_tokens = extra_embed_tokens,
        video_lens = torch.randint(5, 16, (2,)),
        rewards = torch.randn((2, 16))
    )

    loss.backward()

    pred = reward_model(commands, video)

    assert pred.shape == (2, 16)

def test_rewind_wrapper():
    from rewind_reward_pytorch.rewind_reward import RewardModel, RewindTrainWrapper

    reward_model = RewardModel(
        categorical_rewards = False
    )

    commands = [
      'pick up the blue ball and put it in the red tray',
      'pick up the red cube and put it in the green bin',
    ]

    video = torch.rand(2, 3, 16, 224, 224)

    train_wrapper = RewindTrainWrapper(reward_model)

    loss = train_wrapper(
        commands,
        video,
        video_lens = torch.randint(5, 16, (2,)),
    )

    loss.backward()

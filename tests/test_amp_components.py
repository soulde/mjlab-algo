import torch

from mmrl.memories import AMPTransitionMemory, TensorAMPDataset
from mmrl.models import AMPDiscriminator, RunningMeanStd


def test_amp_transition_memory_and_tensor_dataset():
    memory = AMPTransitionMemory(capacity=4, obs_dim=3, device="cpu")
    state = torch.arange(18, dtype=torch.float32).reshape(6, 3)
    memory.add(state, state + 1)

    assert memory.size == 4
    assert memory.sample(3).state.shape == (3, 3)

    dataset = TensorAMPDataset(state, state + 1)
    assert dataset.observation_dim == 3
    assert dataset.sample(5, "cpu").next_state.shape == (5, 3)


def test_amp_discriminator_reward_penalty_and_normalizer():
    discriminator = AMPDiscriminator(3, (8, 4))
    state = torch.randn(6, 3)
    next_state = torch.randn(6, 3)

    assert discriminator(state, next_state).shape == (6, 1)
    assert discriminator.style_reward(state, next_state, 0.2).min() >= 0
    discriminator.gradient_penalty(state, next_state).backward()

    normalizer = RunningMeanStd(3)
    normalizer.update(state)
    normalized = normalizer.normalize(state)
    assert torch.isfinite(normalized).all()

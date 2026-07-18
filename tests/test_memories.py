import torch
from tensordict import TensorDict

from mmrl.memories.off_policy import OffPolicyBatch, OffPolicyReplayMemory
from mmrl.memories.on_policy import OnPolicyRolloutBatch, OnPolicyRolloutMemory
from mmrl.memories.episode import EpisodeMemory


def test_off_policy_memory_samples_transitions():
    memory = OffPolicyReplayMemory(capacity=4, obs_dim=2, action_dim=1, device="cpu")
    memory.add(
        obs=torch.zeros(2, 2),
        action=torch.zeros(2, 1),
        reward=torch.ones(2),
        next_obs=torch.ones(2, 2),
        done=torch.zeros(2, dtype=torch.bool),
    )

    batch = memory.sample(2)
    assert isinstance(batch, OffPolicyBatch)
    assert batch.obs.shape == (2, 2)
    assert batch.action.shape == (2, 1)
    assert batch.reward.shape == (2, 1)
    assert memory.size == 2


def test_off_policy_memory_overwrites_oldest_ring_entries():
    memory = OffPolicyReplayMemory(capacity=3, obs_dim=1, action_dim=1, device="cpu")
    memory.add(
        obs=torch.arange(5, dtype=torch.float32).view(5, 1),
        action=torch.zeros(5, 1),
        reward=torch.zeros(5),
        next_obs=torch.zeros(5, 1),
        done=torch.zeros(5, dtype=torch.bool),
    )

    assert memory.size == 3
    assert sorted(memory.obs.flatten().tolist()) == [2.0, 3.0, 4.0]


class _EpisodeCfg:
    buffer_size = 5
    steps = 10
    batch_size = 1
    horizon = 1


def test_episode_memory_evicts_by_timestep_capacity():
    memory = EpisodeMemory(_EpisodeCfg())
    episode_a = TensorDict({"obs": torch.zeros(3, 1)}, batch_size=(3,))
    episode_b = TensorDict({"obs": torch.ones(3, 1)}, batch_size=(3,))

    assert memory.add(episode_a) == 1
    assert memory.add(episode_b) == 1
    assert memory.size == 3
    assert memory.num_eps == 1


def test_on_policy_rollout_memory_uses_shared_storage():
    memory = OnPolicyRolloutMemory()
    batch = OnPolicyRolloutBatch(
        obs=torch.zeros(1, 2),
        action=torch.zeros(1, 1),
        reward=torch.ones(1),
        done=torch.zeros(1, dtype=torch.bool),
    )

    memory.add(batch)
    assert memory.size == 1
    assert memory.sample() == [batch]
    memory.clear()
    assert memory.size == 0

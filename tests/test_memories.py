import torch

from mmrl.fastsac.buffer import FastSACBatch, FastSACReplayBuffer
from mmrl.memories.off_policy import OffPolicyBatch, OffPolicyReplayMemory
from mmrl.tdmpc2.buffer import Buffer
from mmrl.memories.episode import EpisodeMemory


def test_fastsac_replay_buffer_aliases_off_policy_memory():
    assert FastSACBatch is OffPolicyBatch
    assert FastSACReplayBuffer is OffPolicyReplayMemory

    memory = FastSACReplayBuffer(capacity=4, obs_dim=2, action_dim=1, device="cpu")
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


def test_tdmpc2_buffer_aliases_episode_memory():
    assert Buffer is EpisodeMemory


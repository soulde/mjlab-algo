"""Off-policy transition replay memories."""

from dataclasses import dataclass

import torch

from mmrl.memories.base import Memory
from mmrl.memories.storage import TensorRingStorage


@dataclass(frozen=True)
class OffPolicyBatch:
    obs: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    next_obs: torch.Tensor
    done: torch.Tensor


class OffPolicyReplayMemory(Memory):
    """Fixed-size replay memory storing vectorized transitions."""

    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        action_dim: int,
        device: str | torch.device,
    ):
        self.capacity = int(capacity)
        self.device = torch.device(device)
        self.storage = TensorRingStorage(
            self.capacity,
            {
                "obs": ((obs_dim,), torch.float32),
                "action": ((action_dim,), torch.float32),
                "reward": ((1,), torch.float32),
                "next_obs": ((obs_dim,), torch.float32),
                "done": ((1,), torch.float32),
            },
        )

    @property
    def obs(self) -> torch.Tensor:
        return self.storage["obs"]

    @property
    def action(self) -> torch.Tensor:
        return self.storage["action"]

    @property
    def reward(self) -> torch.Tensor:
        return self.storage["reward"]

    @property
    def next_obs(self) -> torch.Tensor:
        return self.storage["next_obs"]

    @property
    def done(self) -> torch.Tensor:
        return self.storage["done"]

    @property
    def size(self) -> int:
        return self.storage.size

    def add(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        done: torch.Tensor,
    ) -> None:
        """Add a batch of transitions."""
        obs = obs.detach().cpu().to(dtype=torch.float32)
        action = action.detach().cpu().to(dtype=torch.float32)
        reward = reward.detach().cpu().to(dtype=torch.float32).view(-1, 1)
        next_obs = next_obs.detach().cpu().to(dtype=torch.float32)
        done = done.detach().cpu().to(dtype=torch.float32).view(-1, 1)

        self.storage.add(
            {
                "obs": obs,
                "action": action,
                "reward": reward,
                "next_obs": next_obs,
                "done": done,
            }
        )

    def sample(self, batch_size: int) -> OffPolicyBatch:
        """Sample a uniformly random batch."""
        indices = self.storage.sample_indices(batch_size)
        fields = self.storage.gather(indices, device=self.device)
        return OffPolicyBatch(
            obs=fields["obs"],
            action=fields["action"],
            reward=fields["reward"],
            next_obs=fields["next_obs"],
            done=fields["done"],
        )

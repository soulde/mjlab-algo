"""Replay buffer for FastSAC."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class FastSACBatch:
    obs: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    next_obs: torch.Tensor
    done: torch.Tensor


class FastSACReplayBuffer:
    """Fixed-size replay buffer storing vectorized transitions."""

    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        action_dim: int,
        device: str | torch.device,
    ):
        self.capacity = int(capacity)
        self.device = torch.device(device)
        self.obs = torch.empty((self.capacity, obs_dim), dtype=torch.float32)
        self.action = torch.empty((self.capacity, action_dim), dtype=torch.float32)
        self.reward = torch.empty((self.capacity, 1), dtype=torch.float32)
        self.next_obs = torch.empty((self.capacity, obs_dim), dtype=torch.float32)
        self.done = torch.empty((self.capacity, 1), dtype=torch.float32)
        self._pos = 0
        self._size = 0

    @property
    def size(self) -> int:
        return self._size

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

        batch_size = obs.shape[0]
        indices = (torch.arange(batch_size) + self._pos) % self.capacity
        self.obs[indices] = obs
        self.action[indices] = action
        self.reward[indices] = reward
        self.next_obs[indices] = next_obs
        self.done[indices] = done
        self._pos = int((self._pos + batch_size) % self.capacity)
        self._size = min(self._size + batch_size, self.capacity)

    def sample(self, batch_size: int) -> FastSACBatch:
        """Sample a uniformly random batch."""
        if self._size < batch_size:
            raise ValueError(
                f"Cannot sample {batch_size} transitions from {self._size}."
            )
        indices = torch.randint(0, self._size, (batch_size,))
        return FastSACBatch(
            obs=self.obs[indices].to(self.device),
            action=self.action[indices].to(self.device),
            reward=self.reward[indices].to(self.device),
            next_obs=self.next_obs[indices].to(self.device),
            done=self.done[indices].to(self.device),
        )

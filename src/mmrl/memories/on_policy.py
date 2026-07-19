"""On-policy rollout memory with GAE support."""

from dataclasses import dataclass

import torch

from mmrl.memories.base import Memory
from mmrl.memories.storage import TensorRolloutStorage


@dataclass(frozen=True)
class OnPolicyRolloutBatch:
    obs: torch.Tensor
    critic_obs: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    done: torch.Tensor
    log_prob: torch.Tensor
    value: torch.Tensor
    advantage: torch.Tensor
    ret: torch.Tensor


class OnPolicyRolloutMemory(Memory):
    """Preallocated vector rollout storage for PPO-style algorithms."""

    def __init__(
        self,
        num_steps: int,
        num_envs: int,
        obs_shape: tuple[int, ...],
        action_shape: tuple[int, ...],
        device: str | torch.device,
        critic_obs_shape: tuple[int, ...] | None = None,
    ):
        self.storage = TensorRolloutStorage(
            num_steps,
            num_envs,
            {
                "obs": (obs_shape, torch.float32),
                "critic_obs": (critic_obs_shape or obs_shape, torch.float32),
                "action": (action_shape, torch.float32),
                "reward": ((1,), torch.float32),
                "done": ((1,), torch.bool),
                "log_prob": ((1,), torch.float32),
                "value": ((1,), torch.float32),
                "advantage": ((1,), torch.float32),
                "ret": ((1,), torch.float32),
            },
            device,
        )

    @property
    def size(self) -> int:
        return self.storage.size

    @property
    def full(self) -> bool:
        return self.storage.full

    def add(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        log_prob: torch.Tensor,
        value: torch.Tensor,
        critic_obs: torch.Tensor | None = None,
    ) -> None:
        self.storage.add(
            {
                "obs": obs,
                "critic_obs": critic_obs if critic_obs is not None else obs,
                "action": action,
                "reward": reward.reshape(-1, 1),
                "done": done.reshape(-1, 1),
                "log_prob": log_prob.reshape(-1, 1),
                "value": value.reshape(-1, 1),
            }
        )

    def compute_returns(
        self,
        last_value: torch.Tensor,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        normalize_advantage: bool = True,
    ) -> None:
        """Compute generalized advantage estimates for the completed rollout."""
        if not self.full:
            raise RuntimeError("Returns require a complete rollout.")
        advantage = torch.zeros_like(last_value).reshape(-1, 1)
        next_value = last_value.reshape(-1, 1).to(self.storage.device)
        for step in range(self.storage.num_steps - 1, -1, -1):
            not_done = (~self.storage["done"][step]).float()
            delta = (
                self.storage["reward"][step]
                + gamma * next_value * not_done
                - self.storage["value"][step]
            )
            advantage = delta + gamma * gae_lambda * not_done * advantage
            self.storage["advantage"][step].copy_(advantage)
            next_value = self.storage["value"][step]
        self.storage["ret"].copy_(
            self.storage["advantage"] + self.storage["value"]
        )
        if normalize_advantage:
            values = self.storage["advantage"]
            values.sub_(values.mean()).div_(values.std(unbiased=False).clamp_min(1e-8))

    def mini_batches(self, num_mini_batches: int):
        """Yield shuffled, equally sized mini-batches from a full rollout."""
        if not self.full:
            raise RuntimeError("Mini-batches require a complete rollout.")
        batch_size = self.size
        if batch_size % num_mini_batches != 0:
            raise ValueError("Rollout size must be divisible by num_mini_batches.")
        indices = torch.randperm(batch_size, device=self.storage.device)
        for chunk in indices.chunk(num_mini_batches):
            fields = {
                name: self.storage.flatten(name)[chunk]
                for name in self.storage.fields
            }
            yield OnPolicyRolloutBatch(**fields)

    def sample(self, num_mini_batches: int):
        """Return the mini-batch iterator required by the memory contract."""
        return self.mini_batches(num_mini_batches)

    def clear(self) -> None:
        self.storage.clear()

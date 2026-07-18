"""On-policy rollout memories."""

from dataclasses import dataclass

import torch

from mmrl.memories.base import Memory
from mmrl.memories.storage import TensorListStorage


@dataclass(frozen=True)
class OnPolicyRolloutBatch:
    obs: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    done: torch.Tensor
    log_prob: torch.Tensor | None = None
    value: torch.Tensor | None = None
    advantage: torch.Tensor | None = None
    ret: torch.Tensor | None = None


class OnPolicyRolloutMemory(Memory):
    """Minimal rollout store for future on-policy algorithms."""

    def __init__(self):
        self.storage = TensorListStorage()

    @property
    def size(self) -> int:
        return self.storage.size

    def add(self, batch: OnPolicyRolloutBatch) -> None:
        self.storage.add(batch)

    def clear(self) -> None:
        self.storage.clear()

    def sample(self) -> list[OnPolicyRolloutBatch]:
        return self.storage.as_list()

"""On-policy rollout memories."""

from dataclasses import dataclass

import torch

from mmrl.memories.base import Memory


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
        self._items: list[OnPolicyRolloutBatch] = []

    @property
    def size(self) -> int:
        return len(self._items)

    def add(self, batch: OnPolicyRolloutBatch) -> None:
        self._items.append(batch)

    def clear(self) -> None:
        self._items.clear()

    def sample(self) -> list[OnPolicyRolloutBatch]:
        return list(self._items)


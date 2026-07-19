"""Transition replay and expert data contracts for AMP."""

from dataclasses import dataclass
from typing import Protocol

import torch

from mmrl.memories.base import Memory
from mmrl.memories.storage import TensorRingStorage


@dataclass(frozen=True)
class AMPTransitionBatch:
    state: torch.Tensor
    next_state: torch.Tensor


class AMPExpertSource(Protocol):
    """Environment-owned source of consecutive expert AMP observations."""

    @property
    def observation_dim(self) -> int: ...

    def sample(
        self, batch_size: int, device: str | torch.device
    ) -> AMPTransitionBatch: ...


class AMPTransitionMemory(Memory):
    """Device-resident ring replay for policy AMP transitions."""

    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        device: str | torch.device,
    ) -> None:
        self.device = torch.device(device)
        self.storage = TensorRingStorage(
            capacity,
            {
                "state": ((obs_dim,), torch.float32),
                "next_state": ((obs_dim,), torch.float32),
            },
            self.device,
        )

    @property
    def size(self) -> int:
        return self.storage.size

    def add(self, state: torch.Tensor, next_state: torch.Tensor) -> None:
        self.storage.add(
            {
                "state": state.detach().to(dtype=torch.float32),
                "next_state": next_state.detach().to(dtype=torch.float32),
            }
        )

    def sample(self, batch_size: int) -> AMPTransitionBatch:
        indices = self.storage.sample_indices(batch_size)
        values = self.storage.gather(indices, device=self.device)
        return AMPTransitionBatch(**values)


class TensorAMPDataset:
    """In-memory expert source for preprocessed AMP transition tensors."""

    def __init__(self, state: torch.Tensor, next_state: torch.Tensor) -> None:
        if state.ndim != 2 or state.shape != next_state.shape:
            raise ValueError("AMP expert tensors must have matching (N, D) shapes.")
        if state.shape[0] == 0:
            raise ValueError("AMP expert dataset cannot be empty.")
        self.state = state.detach().to("cpu", dtype=torch.float32)
        self.next_state = next_state.detach().to("cpu", dtype=torch.float32)

    @property
    def observation_dim(self) -> int:
        return self.state.shape[1]

    def sample(
        self, batch_size: int, device: str | torch.device
    ) -> AMPTransitionBatch:
        indices = torch.randint(0, self.state.shape[0], (batch_size,))
        return AMPTransitionBatch(
            self.state[indices].to(device),
            self.next_state[indices].to(device),
        )

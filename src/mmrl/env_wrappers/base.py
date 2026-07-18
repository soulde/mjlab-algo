"""Common environment wrapper interfaces."""

from abc import ABC, abstractmethod
from typing import Any

import torch


class EnvWrapper(ABC):
    """Minimal environment contract consumed by algorithm runners."""

    @property
    @abstractmethod
    def num_envs(self) -> int:
        """Number of parallel environment instances."""

    @property
    @abstractmethod
    def obs_dim(self) -> int:
        """Flattened observation dimension."""

    @property
    @abstractmethod
    def action_dim(self) -> int:
        """Flattened action dimension."""

    @property
    @abstractmethod
    def device(self) -> torch.device:
        """Device used for environment tensors."""

    @property
    @abstractmethod
    def unwrapped(self) -> Any:
        """Return the underlying raw environment."""

    @abstractmethod
    def rand_act(self) -> torch.Tensor:
        """Return a random action tensor compatible with ``step``."""

    @abstractmethod
    def reset(self) -> torch.Tensor:
        """Reset the environment and return a flattened observation tensor."""

    @abstractmethod
    def step(self, action: torch.Tensor):
        """Step the environment."""

    @abstractmethod
    def close(self) -> None:
        """Close the environment."""


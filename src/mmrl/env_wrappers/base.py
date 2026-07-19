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

    def get_amp_observations(self) -> torch.Tensor:
        """Return simulator-specific AMP observations through a common API."""
        environment = self.unwrapped
        getter = getattr(environment, "get_amp_observations", None)
        if not callable(getter):
            raise TypeError(
                "The wrapped environment must implement get_amp_observations()."
            )
        return self._format_amp_observations(getter())

    @property
    def amp_observation_dim(self) -> int:
        """Dimension of one simulator AMP observation."""
        return int(self.get_amp_observations().shape[-1])

    def _format_amp_observations(self, value: Any) -> torch.Tensor:
        tensor = torch.as_tensor(value, dtype=torch.float32, device=self.device)
        if tensor.ndim == 1 and self.num_envs == 1:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 2 or tensor.shape[0] != self.num_envs:
            raise ValueError(
                "AMP observations must have shape (num_envs, amp_obs_dim)."
            )
        return tensor

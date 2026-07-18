"""Base runner contracts."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable

import torch


class Runner(ABC):
    """Base class for training/evaluation loops."""

    @abstractmethod
    def learn(self) -> None:
        """Run training."""

    @abstractmethod
    def get_inference_policy(
        self, device: str | torch.device | None = None
    ) -> Callable:
        """Return the callable used by environment play scripts."""

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """Save an algorithm checkpoint."""

    @abstractmethod
    def load(self, path: str | Path) -> None:
        """Load an algorithm checkpoint."""

"""Base runner contracts."""

from abc import ABC, abstractmethod


class Runner(ABC):
    """Base class for training/evaluation loops."""

    @abstractmethod
    def train(self) -> None:
        """Run training."""


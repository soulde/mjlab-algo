"""Common memory interfaces."""

from abc import ABC, abstractmethod


class Memory(ABC):
    """Base class for algorithm data stores."""

    @property
    @abstractmethod
    def size(self) -> int:
        """Number of stored transitions or timesteps."""

    @abstractmethod
    def sample(self, *args, **kwargs):
        """Sample training data."""


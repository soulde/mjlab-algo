"""Shared storage backends for memories."""

from collections.abc import Mapping

import torch


class TensorRingStorage:
    """Preallocated ring storage for fixed-shape tensor fields."""

    def __init__(
        self,
        capacity: int,
        specs: Mapping[str, tuple[tuple[int, ...], torch.dtype]],
    ):
        self.capacity = int(capacity)
        self.fields = {
            name: torch.empty((self.capacity, *shape), dtype=dtype)
            for name, (shape, dtype) in specs.items()
        }
        self._pos = 0
        self._size = 0

    @property
    def size(self) -> int:
        return self._size

    def __getitem__(self, field: str) -> torch.Tensor:
        return self.fields[field]

    def add(self, values: Mapping[str, torch.Tensor]) -> None:
        """Add a batch of field values to the ring."""
        if not values:
            return
        batch_size = next(iter(values.values())).shape[0]
        if batch_size > self.capacity:
            start = batch_size - self.capacity
            values = {name: value[start:] for name, value in values.items()}
            batch_size = self.capacity

        indices = (torch.arange(batch_size) + self._pos) % self.capacity
        for name, value in values.items():
            self.fields[name][indices] = value

        self._pos = int((self._pos + batch_size) % self.capacity)
        self._size = min(self._size + batch_size, self.capacity)

    def sample_indices(self, batch_size: int) -> torch.Tensor:
        if self._size < batch_size:
            raise ValueError(
                f"Cannot sample {batch_size} transitions from {self._size}."
            )
        return torch.randint(0, self._size, (batch_size,))

    def gather(
        self,
        indices: torch.Tensor,
        fields: tuple[str, ...] | None = None,
        device: str | torch.device | None = None,
    ) -> dict[str, torch.Tensor]:
        target = torch.device(device) if device is not None else None
        names = fields or tuple(self.fields)
        result = {}
        for name in names:
            value = self.fields[name][indices]
            result[name] = value.to(target) if target is not None else value
        return result


class EpisodeListStorage:
    """Capacity-limited storage for variable-length episodes."""

    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.episodes: list = []
        self._num_steps = 0

    @property
    def size(self) -> int:
        return self._num_steps

    @property
    def num_eps(self) -> int:
        return len(self.episodes)

    def add(self, episode) -> int:
        episode_len = int(episode.shape[0])
        while self._num_steps + episode_len > self.capacity and self.episodes:
            removed = self.episodes.pop(0)
            self._num_steps -= int(removed.shape[0])

        self.episodes.append(episode)
        self._num_steps += episode_len
        return self.num_eps

    def transition_counts(self, min_len: int) -> torch.Tensor:
        return torch.tensor(
            [max(0, int(ep.shape[0]) - min_len + 1) for ep in self.episodes],
            dtype=torch.float32,
        )

    def sample_episode(self, min_len: int):
        lengths = self.transition_counts(min_len)
        if lengths.numel() == 0 or float(lengths.sum()) <= 0.0:
            raise ValueError("Cannot sample: no episode is long enough.")
        probs = lengths / lengths.sum()
        idx = torch.multinomial(probs, 1).item()
        return self.episodes[idx]


class TensorRolloutStorage:
    """Device-resident preallocated storage for vectorized rollouts."""

    def __init__(
        self,
        num_steps: int,
        num_envs: int,
        specs: Mapping[str, tuple[tuple[int, ...], torch.dtype]],
        device: str | torch.device,
    ):
        self.num_steps = int(num_steps)
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.fields = {
            name: torch.empty(
                (self.num_steps, self.num_envs, *shape),
                dtype=dtype,
                device=self.device,
            )
            for name, (shape, dtype) in specs.items()
        }
        self._pos = 0

    @property
    def size(self) -> int:
        return self._pos * self.num_envs

    @property
    def full(self) -> bool:
        return self._pos == self.num_steps

    def __getitem__(self, field: str) -> torch.Tensor:
        return self.fields[field]

    def add(self, values: Mapping[str, torch.Tensor]) -> None:
        if self.full:
            raise RuntimeError("Rollout storage is full; clear it before adding.")
        for name, value in values.items():
            self.fields[name][self._pos].copy_(value.to(self.device))
        self._pos += 1

    def clear(self) -> None:
        self._pos = 0

    def flatten(self, field: str) -> torch.Tensor:
        value = self.fields[field][0 : self._pos]
        return value.flatten(0, 1)

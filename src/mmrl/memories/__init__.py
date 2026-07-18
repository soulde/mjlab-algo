"""Memory buffers used by runners and agents."""

from mmrl.memories.base import Memory as Memory
from mmrl.memories.off_policy import (
    OffPolicyBatch as OffPolicyBatch,
    OffPolicyReplayMemory as OffPolicyReplayMemory,
)
from mmrl.memories.on_policy import (
    OnPolicyRolloutBatch as OnPolicyRolloutBatch,
    OnPolicyRolloutMemory as OnPolicyRolloutMemory,
)
from mmrl.memories.storage import (
    EpisodeListStorage as EpisodeListStorage,
    TensorListStorage as TensorListStorage,
    TensorRingStorage as TensorRingStorage,
)

__all__ = [
    "EpisodeMemory",
    "Memory",
    "OffPolicyBatch",
    "OffPolicyReplayMemory",
    "OnPolicyRolloutBatch",
    "OnPolicyRolloutMemory",
    "EpisodeListStorage",
    "TensorListStorage",
    "TensorRingStorage",
]


def __getattr__(name: str):
    if name == "EpisodeMemory":
        from mmrl.memories.episode import EpisodeMemory

        return EpisodeMemory
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

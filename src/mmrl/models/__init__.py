"""Reusable model building blocks."""

from mmrl.models.actors import (
    SquashedGaussianActor as SquashedGaussianActor,
    init_weights as init_weights,
)
from mmrl.models.critics import TwinQNetwork as TwinQNetwork

__all__ = [
    "SquashedGaussianActor",
    "TwinQNetwork",
    "WorldModel",
    "init_weights",
]


def __getattr__(name: str):
    if name == "WorldModel":
        from mmrl.models.world_models import WorldModel

        return WorldModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

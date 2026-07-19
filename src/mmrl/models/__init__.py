"""Reusable model building blocks."""

from mmrl.models.amp import AMPDiscriminator as AMPDiscriminator
from mmrl.models.amp import RunningMeanStd as RunningMeanStd
from mmrl.models.base import Model as Model
from mmrl.models.actors import (
    GaussianActor as GaussianActor,
    SquashedGaussianActor as SquashedGaussianActor,
    init_weights as init_weights,
)
from mmrl.models.critics import (
    QNetwork as QNetwork,
    TwinQNetwork as TwinQNetwork,
    ValueNetwork as ValueNetwork,
)

__all__ = [
    "GaussianActor",
    "AMPDiscriminator",
    "Model",
    "QNetwork",
    "SquashedGaussianActor",
    "TwinQNetwork",
    "ValueNetwork",
    "WorldModel",
    "RunningMeanStd",
    "init_weights",
]


def __getattr__(name: str):
    if name == "WorldModel":
        from mmrl.models.world_models import WorldModel

        return WorldModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

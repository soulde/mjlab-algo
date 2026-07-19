"""Adversarial Motion Priors."""

from mmrl.amp.config import (
    AMPAlgorithmCfg as AMPAlgorithmCfg,
    AMPDiscriminatorCfg as AMPDiscriminatorCfg,
    AMPMemoryCfg as AMPMemoryCfg,
    AMPRunnerCfg as AMPRunnerCfg,
)
from mmrl.amp.motion_loader import AMPLoader as AMPLoader

__all__ = [
    "AMP",
    "AMPAlgorithmCfg",
    "AMPLoader",
    "AMPDiscriminatorCfg",
    "AMPMemoryCfg",
    "AMPRunner",
    "AMPRunnerCfg",
]


def __getattr__(name: str):
    if name == "AMP":
        from mmrl.amp.amp import AMP

        return AMP
    if name == "AMPRunner":
        from mmrl.amp.runner import AMPRunner

        return AMPRunner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

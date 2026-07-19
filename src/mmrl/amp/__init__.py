"""Adversarial Motion Priors."""

from mmrl.amp.config import (
    AMPAlgorithmCfg as AMPAlgorithmCfg,
    AMPDiscriminatorCfg as AMPDiscriminatorCfg,
    AMPMemoryCfg as AMPMemoryCfg,
    AMPRunnerCfg as AMPRunnerCfg,
)

__all__ = [
    "AMP",
    "AMPAlgorithmCfg",
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

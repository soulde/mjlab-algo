"""TD-MPC2 algorithm package."""

from mmrl.tdmpc2.config import (
    MODEL_SIZE as MODEL_SIZE,
    EpisodeMemoryCfg as EpisodeMemoryCfg,
    TDMPC2AlgorithmCfg as TDMPC2AlgorithmCfg,
    TDMPC2ModelCfg as TDMPC2ModelCfg,
    TDMPC2RunnerCfg as TDMPC2RunnerCfg,
)

__all__ = [
    "MODEL_SIZE",
    "EpisodeMemoryCfg",
    "TDMPC2",
    "TDMPC2AlgorithmCfg",
    "TDMPC2ModelCfg",
    "TDMPC2Runner",
    "TDMPC2RunnerCfg",
]


def __getattr__(name: str):
    if name == "TDMPC2":
        from mmrl.tdmpc2.tdmpc2 import TDMPC2

        return TDMPC2
    if name == "TDMPC2Runner":
        from mmrl.tdmpc2.runner import TDMPC2Runner

        return TDMPC2Runner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

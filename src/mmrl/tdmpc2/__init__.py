"""TD-MPC2 algorithm package."""

from mmrl.tdmpc2.config import (
    MODEL_SIZE as MODEL_SIZE,
    TDMPC2Config as TDMPC2Config,
    make_tdmpc2_config as make_tdmpc2_config,
)

__all__ = [
    "MODEL_SIZE",
    "TDMPC2",
    "TDMPC2Config",
    "TDMPC2Runner",
    "make_tdmpc2_config",
]


def __getattr__(name: str):
    if name == "TDMPC2":
        from mmrl.tdmpc2.tdmpc2 import TDMPC2

        return TDMPC2
    if name == "TDMPC2Runner":
        from mmrl.tdmpc2.runner import TDMPC2Runner

        return TDMPC2Runner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

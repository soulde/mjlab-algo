"""TD-MPC2 algorithm package."""

# ruff: noqa: E402

from mjlab_algo.tdmpc2.compile import (
    configure_tdmpc2_compile as configure_tdmpc2_compile,
)

configure_tdmpc2_compile(enabled=True)

from mjlab_algo.tdmpc2.buffer import Buffer as Buffer
from mjlab_algo.tdmpc2.config import (
    MODEL_SIZE as MODEL_SIZE,
    TASK_SET as TASK_SET,
    TDMPC2Config as TDMPC2Config,
    make_tdmpc2_config as make_tdmpc2_config,
)
from mjlab_algo.tdmpc2.runner import TDMPC2Runner as TDMPC2Runner
from mjlab_algo.tdmpc2.tdmpc2 import TDMPC2 as TDMPC2

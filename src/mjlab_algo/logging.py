"""Console logging helpers for MJLab algorithm runners."""

from __future__ import annotations

import datetime
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch


def scalar(value: Any) -> float:
    """Convert tensors and numeric values to a Python float for logging."""
    if isinstance(value, torch.Tensor):
        return float(value.detach().mean().cpu())
    if isinstance(value, (int, float)):
        return float(value)
    return float(value)


def format_duration(seconds: float) -> str:
    """Format seconds as a compact timedelta string."""
    return str(datetime.timedelta(seconds=int(max(seconds, 0.0))))


def format_training_log(
    *,
    title: str,
    total_steps: int,
    steps_per_second: float,
    collection_time: float,
    learning_time: float,
    losses: Mapping[str, Any] | None = None,
    mean_reward: float | None = None,
    mean_episode_length: float | None = None,
    extras: Mapping[str, Any] | None = None,
    iteration_time: float,
    elapsed_time: float,
    eta_seconds: float,
    log_dir: str | Path | None = None,
    width: int = 80,
    pad: int = 40,
) -> str:
    """Build a PPO-style console log block."""
    log_string = f"{'#' * width}\n"
    log_string += f"\033[1m{f' {title} '.center(width)}\033[0m \n\n"
    if log_dir is not None:
        log_string += f"{'Log directory:':>{pad}} {log_dir}\n"
    log_string += (
        f"{'Total steps:':>{pad}} {total_steps} \n"
        f"{'Steps per second:':>{pad}} {steps_per_second:.0f} \n"
        f"{'Collection time:':>{pad}} {collection_time:.3f}s \n"
        f"{'Learning time:':>{pad}} {learning_time:.3f}s \n"
    )
    for name, value in (losses or {}).items():
        log_string += f"{f'Mean {name} loss:':>{pad}} {scalar(value):.4f}\n"
    if mean_reward is not None:
        log_string += f"{'Mean reward:':>{pad}} {mean_reward:.2f}\n"
    if mean_episode_length is not None:
        log_string += f"{'Mean episode length:':>{pad}} {mean_episode_length:.2f}\n"
    for name, value in (extras or {}).items():
        log_string += f"{f'{name}:':>{pad}} {scalar(value):.4f}\n"
    log_string += (
        f"{'-' * width}\n"
        f"{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"
        f"{'Time elapsed:':>{pad}} {format_duration(elapsed_time)}\n"
        f"{'ETA:':>{pad}} {format_duration(eta_seconds)}\n"
    )
    return log_string

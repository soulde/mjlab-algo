"""Logging helpers shared by all algorithm runners."""

from __future__ import annotations

import datetime
import importlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass
class LoggerCfg:
    """Configuration for optional metric logging backends."""

    backends: tuple[str, ...] = ()
    color: bool = True
    wandb_project: str = "mmrl"
    wandb_entity: str | None = None
    wandb_group: str | None = None
    run_name: str | None = None
    wandb_silent: bool = False


class MetricLogger:
    """Fan scalar metrics out to configured experiment tracking backends."""

    _SUPPORTED_BACKENDS = frozenset({"tensorboard", "wandb"})

    def __init__(
        self,
        log_dir: str | Path,
        cfg: LoggerCfg | Mapping[str, Any] | Any | None = None,
        run_config: Mapping[str, Any] | None = None,
    ) -> None:
        self.log_dir = Path(log_dir)
        self.cfg = cfg or LoggerCfg()
        self._tensorboard = None
        self._wandb_run = None

        backends = tuple(dict.fromkeys(_logger_cfg_value(self.cfg, "backends", ())))
        unsupported = set(backends) - self._SUPPORTED_BACKENDS
        if unsupported:
            names = ", ".join(sorted(unsupported))
            raise ValueError(f"Unsupported logging backend(s): {names}")
        if backends:
            self.log_dir.mkdir(parents=True, exist_ok=True)

        if "tensorboard" in backends:
            try:
                summary_writer = importlib.import_module(
                    "torch.utils.tensorboard"
                ).SummaryWriter
            except ImportError as exc:
                raise ImportError(
                    "The required `tensorboard` package is not installed."
                ) from exc
            self._tensorboard = summary_writer(log_dir=str(self.log_dir))

        if "wandb" in backends:
            try:
                wandb = importlib.import_module("wandb")
            except ImportError as exc:
                raise ImportError(
                    "The required `wandb` package is not installed."
                ) from exc
            self._wandb_run = wandb.init(
                project=_logger_cfg_value(self.cfg, "wandb_project", "mmrl"),
                entity=_logger_cfg_value(self.cfg, "wandb_entity"),
                group=_logger_cfg_value(self.cfg, "wandb_group"),
                name=_logger_cfg_value(self.cfg, "run_name"),
                dir=str(self.log_dir),
                config=dict(run_config or {}),
                settings=wandb.Settings(silent=True)
                if _logger_cfg_value(self.cfg, "wandb_silent", False)
                else None,
            )

    def log(
        self,
        metrics: Mapping[str, Any],
        step: int,
        prefix: str | None = None,
    ) -> None:
        """Record scalar metrics at a global training step."""
        values = {
            f"{prefix}/{name}" if prefix else name: scalar(value)
            for name, value in metrics.items()
        }
        if self._tensorboard is not None:
            for name, value in values.items():
                self._tensorboard.add_scalar(name, value, step)
        if self._wandb_run is not None:
            self._wandb_run.log(values, step=step)

    def close(self) -> None:
        """Flush and close all active logging backends."""
        if self._tensorboard is not None:
            self._tensorboard.close()
            self._tensorboard = None
        if self._wandb_run is not None:
            self._wandb_run.finish()
            self._wandb_run = None


def _logger_cfg_value(cfg: Any, name: str, default: Any = None) -> Any:
    if isinstance(cfg, Mapping):
        return cfg.get(name, default)
    return getattr(cfg, name, default)


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
    color: bool = True,
    width: int = 80,
    pad: int = 40,
) -> str:
    """Build a PPO-style console log block."""
    def paint(text: str, code: str) -> str:
        return f"\033[{code}m{text}\033[0m" if color else text

    def row(label: str, value: str, code: str) -> str:
        return f"{label:>{pad}} {paint(value, code)}\n"

    log_string = f"{paint('#' * width, '90')}\n"
    log_string += f"{paint(f' {title} '.center(width), '1;36')} \n\n"
    if log_dir is not None:
        log_string += row("Log directory:", str(log_dir), "36")
    log_string += (
        row("Total steps:", str(total_steps), "1;37")
        + row("Steps per second:", f"{steps_per_second:.0f}", "1;32")
        + row("Collection time:", f"{collection_time:.3f}s", "32")
        + row("Learning time:", f"{learning_time:.3f}s", "32")
    )
    for name, value in (losses or {}).items():
        log_string += row(f"Mean {name} loss:", f"{scalar(value):.4f}", "33")
    if mean_reward is not None:
        log_string += row("Mean reward:", f"{mean_reward:.2f}", "1;35")
    if mean_episode_length is not None:
        log_string += row(
            "Mean episode length:", f"{mean_episode_length:.2f}", "35"
        )
    for name, value in (extras or {}).items():
        log_string += row(f"{name}:", f"{scalar(value):.4f}", "34")
    log_string += (
        f"{paint('-' * width, '90')}\n"
        + row("Iteration time:", f"{iteration_time:.2f}s", "36")
        + row("Time elapsed:", format_duration(elapsed_time), "36")
        + row("ETA:", format_duration(eta_seconds), "1;36")
    )
    return log_string

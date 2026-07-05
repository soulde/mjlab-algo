"""Training script for TD-MPC2.

CLI-driven training entry point using tyro.
"""

import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab_algo.tdmpc2 import (
    TDMPC2,
    Buffer,
    TDMPC2Runner,
    make_tdmpc2_config,
)
from mjlab_algo.tdmpc2.vecenv_wrapper import TDMPC2VecEnvWrapper
from mjlab_algo.scripts._cli import maybe_print_top_level_help
from mjlab.tasks.registry import list_tasks, load_env_cfg
from mjlab.utils.torch import configure_torch_backends


@dataclass(frozen=True)
class TrainArgs:
    """CLI arguments for TD-MPC2 training."""

    task_id: str = ""
    """Task ID (e.g., ``Mjlab-Cartpole-Balance``)."""
    model_size: int | None = 5
    """Model size preset: 1, 5, 19, 48, or 317."""
    steps: int = 1_000_000
    """Total environment steps."""
    seed: int = 1
    """Random seed."""
    log_root: str = "logs/tdmpc2"
    """Root directory for logs."""
    exp_name: str = "default"
    """Experiment name."""
    device: str | None = None
    """Device override (default: cuda:0 if available)."""
    mpc: bool = True
    """Whether to use MPPI planning."""
    compile: bool = True
    """Whether to use torch.compile."""
    wandb_project: str = "mjlab"
    """W&B project name."""
    enable_wandb: bool = True
    """Enable W&B logging."""
    save_video: bool = False
    """Record evaluation videos."""
    episodic: bool = False
    """Whether the task has early termination."""
    lr: float = 3e-4
    """Learning rate override."""
    batch_size: int = 256
    """Batch size."""
    buffer_size: int = 1_000_000
    """Replay buffer capacity."""


def main():
    maybe_print_top_level_help("tdmpc2-train")

    import mjlab.tasks  # noqa: F401

    all_tasks = list_tasks()
    if not all_tasks:
        print("No tasks registered. Import mjlab.tasks to populate the registry.")
        sys.exit(1)

    chosen_task, remaining_args = tyro.cli(
        tyro.extras.literal_type_from_choices(all_tasks),
        add_help=False,
        return_unknown_args=True,
    )

    args = tyro.cli(
        TrainArgs,
        args=remaining_args,
        default=TrainArgs(task_id=chosen_task),
        prog=sys.argv[0] + f" {chosen_task}",
    )

    configure_torch_backends()

    # Load environment config from task registry.
    env_cfg = load_env_cfg(args.task_id)

    # Build TD-MPC2 config with model size preset and CLI overrides.
    cfg = make_tdmpc2_config(
        model_size=args.model_size,
        task=args.task_id,
        steps=args.steps,
        seed=args.seed,
        log_root=args.log_root,
        exp_name=args.exp_name,
        mpc=args.mpc,
        compile=args.compile,
        wandb_project=args.wandb_project,
        enable_wandb=args.enable_wandb,
        save_video=args.save_video,
        episodic=args.episodic,
        lr=args.lr,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
    )

    # Set convenience fields.
    cfg.bin_size = (cfg.vmax - cfg.vmin) / (cfg.num_bins - 1)
    cfg.task_title = args.task_id.replace("-", " ").title()
    cfg.multitask = False
    cfg.task_dim = 0

    # Create log directory.
    log_dir = (
        Path(cfg.log_root)
        / cfg.exp_name
        / str(cfg.seed)
        / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    print(f"[INFO] Log directory: {log_dir}")

    # Create environment (use num_envs=1 for TD-MPC2).
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    env_cfg.scene.num_envs = 1
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    env = TDMPC2VecEnvWrapper(env)

    # Populate config with environment info.
    # Calculate total state dim from the DictSpace.
    total_obs_dim = 0
    if hasattr(env.observation_space, "spaces"):
        for space in env.observation_space.spaces.values():
            total_obs_dim += int(space.shape[0])
    else:
        total_obs_dim = int(env.observation_space.shape[0])
    cfg.obs_shape = {"state": (total_obs_dim,)}
    cfg.action_dim = int(env._action_dim)
    cfg.episode_length = env.unwrapped.max_episode_length
    cfg.seed_steps = max(1000, 5 * cfg.episode_length)

    # Seed everything.
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    # Create agent, buffer, and runner.
    agent = TDMPC2(cfg)
    buffer = Buffer(cfg)
    runner = TDMPC2Runner(cfg, env, agent, buffer, log_dir)

    try:
        runner.train()
    finally:
        env.close()


if __name__ == "__main__":
    main()

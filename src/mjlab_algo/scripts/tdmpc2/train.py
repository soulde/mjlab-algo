"""Training script for TD-MPC2.

CLI-driven training entry point using tyro.
"""

import random
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab_algo.tdmpc2 import (
    TDMPC2,
    Buffer,
    MODEL_SIZE,
    TDMPC2Config,
    TDMPC2Runner,
    make_tdmpc2_config,
)
from mjlab_algo.tdmpc2.vecenv_wrapper import TDMPC2VecEnvWrapper
from mjlab_algo.registry import load_tdmpc2_cfg
from mjlab_algo.scripts._cli import maybe_print_top_level_help
from mjlab.tasks.registry import list_tasks, load_env_cfg
from mjlab.utils.torch import configure_torch_backends


def _apply_model_size_from_cfg(cfg: TDMPC2Config) -> TDMPC2Config:
    """Apply a model-size preset after CLI overrides are parsed."""
    if cfg.model_size is None:
        return cfg
    overrides = asdict(cfg)
    model_size = overrides.pop("model_size")
    for field_name in {key for preset in MODEL_SIZE.values() for key in preset}:
        overrides.pop(field_name, None)
    new_cfg = make_tdmpc2_config(model_size=model_size, **overrides)
    new_cfg.model_size = model_size
    return new_cfg


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

    cfg = tyro.cli(
        TDMPC2Config,
        args=remaining_args,
        default=load_tdmpc2_cfg(chosen_task),
        prog=sys.argv[0] + f" {chosen_task}",
    )
    cfg.task = chosen_task
    cfg = _apply_model_size_from_cfg(cfg)

    configure_torch_backends()

    # Load environment config from task registry.
    env_cfg = load_env_cfg(cfg.task)

    # Set convenience fields.
    cfg.bin_size = (cfg.vmax - cfg.vmin) / (cfg.num_bins - 1)
    cfg.task_title = cfg.task.replace("-", " ").title()
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
    device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
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

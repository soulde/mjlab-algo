"""Training script for FastSAC."""

import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import list_tasks, load_env_cfg
from mjlab.utils.torch import configure_torch_backends
from mjlab_algo.fastsac import (
    FastSAC,
    FastSACConfig,
    FastSACReplayBuffer,
    FastSACRunner,
)
from mjlab_algo.fastsac.vecenv_wrapper import FastSACVecEnvWrapper
from mjlab_algo.registry import load_fastsac_cfg
from mjlab_algo.scripts._cli import maybe_print_top_level_help


def launch_training(task_id: str, cfg: FastSACConfig) -> None:
    cfg.task = task_id
    configure_torch_backends()
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    env_cfg = load_env_cfg(task_id)
    env_cfg.scene.num_envs = cfg.num_envs
    env_cfg.seed = cfg.seed
    device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    wrapped_env = FastSACVecEnvWrapper(env)
    obs = wrapped_env.reset()

    cfg.device = device
    cfg.obs_dim = obs.shape[-1]
    cfg.action_dim = wrapped_env.action_dim
    cfg.episode_length = wrapped_env.unwrapped.max_episode_length
    log_dir = (
        Path(cfg.log_root)
        / cfg.exp_name
        / str(cfg.seed)
        / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    print(f"[INFO] Log directory: {log_dir}")

    agent = FastSAC(cfg)
    buffer = FastSACReplayBuffer(
        capacity=cfg.buffer_size,
        obs_dim=cfg.obs_dim,
        action_dim=cfg.action_dim,
        device=agent.device,
    )
    runner = FastSACRunner(cfg, wrapped_env, agent, buffer, log_dir)
    try:
        runner.train()
    finally:
        wrapped_env.close()


def main():
    maybe_print_top_level_help("fastsac-train")

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
        FastSACConfig,
        args=remaining_args,
        default=load_fastsac_cfg(chosen_task),
        prog=sys.argv[0] + f" {chosen_task}",
    )
    launch_training(chosen_task, args)


if __name__ == "__main__":
    main()

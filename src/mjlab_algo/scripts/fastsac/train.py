"""Training script for FastSAC."""

import random
import sys
from dataclasses import dataclass
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
    FastSACReplayBuffer,
    FastSACRunner,
    make_fastsac_config,
)
from mjlab_algo.fastsac.vecenv_wrapper import FastSACVecEnvWrapper
from mjlab_algo.scripts._cli import maybe_print_top_level_help


@dataclass(frozen=True)
class TrainArgs:
    task_id: str = ""
    total_steps: int = 1_000_000
    num_envs: int = 1
    seed: int = 1
    log_root: str = "logs/fastsac"
    exp_name: str = "default"
    device: str | None = None
    batch_size: int = 256
    buffer_size: int = 1_000_000
    learning_starts: int = 5_000
    train_every: int = 1
    gradient_steps: int = 1
    gamma: float = 0.99
    tau: float = 0.005
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    save_agent: bool = True
    save_interval: int = 100_000
    log_interval: int = 1_000


def launch_training(task_id: str, args: TrainArgs) -> None:
    configure_torch_backends()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env_cfg = load_env_cfg(task_id)
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    wrapped_env = FastSACVecEnvWrapper(env)
    obs = wrapped_env.reset()

    cfg = make_fastsac_config(
        task=task_id,
        seed=args.seed,
        total_steps=args.total_steps,
        num_envs=args.num_envs,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        train_every=args.train_every,
        gradient_steps=args.gradient_steps,
        gamma=args.gamma,
        tau=args.tau,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        alpha_lr=args.alpha_lr,
        save_agent=args.save_agent,
        save_interval=args.save_interval,
        log_interval=args.log_interval,
        log_root=args.log_root,
        exp_name=args.exp_name,
        device=device,
        obs_dim=obs.shape[-1],
        action_dim=wrapped_env.action_dim,
        episode_length=wrapped_env.unwrapped.max_episode_length,
    )
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
        TrainArgs,
        args=remaining_args,
        default=TrainArgs(task_id=chosen_task),
        prog=sys.argv[0] + f" {chosen_task}",
    )
    launch_training(chosen_task, args)


if __name__ == "__main__":
    main()

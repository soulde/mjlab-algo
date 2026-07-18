"""Gymnasium FastSAC training example.

Run from an environment with gymnasium installed:

    PYTHONPATH=src python examples/gymnasium/fastsac_train.py Pendulum-v1
"""

import argparse
import random
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from mmrl.env_wrappers.gymnasium import GymnasiumEnvWrapper
from mmrl.fastsac import FastSAC, FastSACConfig, FastSACRunner
from mmrl.memories.off_policy import OffPolicyReplayMemory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("env_id", nargs="?", default="Pendulum-v1")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--total-steps", type=int, default=50_000)
    parser.add_argument("--learning-starts", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--buffer-size", type=int, default=100_000)
    parser.add_argument("--device", default=None)
    parser.add_argument("--log-root", default="logs/gymnasium/fastsac")
    parser.add_argument("--exp-name", default="default")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = gym.make(args.env_id)
    wrapped_env = GymnasiumEnvWrapper(env, device=args.device)
    obs = wrapped_env.reset()

    cfg = FastSACConfig(
        task=args.env_id,
        seed=args.seed,
        total_steps=args.total_steps,
        num_envs=wrapped_env.num_envs,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        device=str(wrapped_env.device),
        obs_dim=obs.shape[-1],
        action_dim=wrapped_env.action_dim,
        exp_name=args.exp_name,
        log_root=args.log_root,
    )
    log_dir = (
        Path(cfg.log_root)
        / cfg.exp_name
        / str(cfg.seed)
        / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )

    agent = FastSAC(cfg)
    memory = OffPolicyReplayMemory(
        capacity=cfg.buffer_size,
        obs_dim=cfg.obs_dim,
        action_dim=cfg.action_dim,
        device=agent.device,
    )
    runner = FastSACRunner(cfg, wrapped_env, agent, memory, log_dir)
    try:
        runner.train()
    finally:
        wrapped_env.close()


if __name__ == "__main__":
    main()


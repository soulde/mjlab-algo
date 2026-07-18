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
from mmrl.fastsac import FastSACRunner, FastSACRunnerCfg, OffPolicyMemoryCfg


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
    cfg = FastSACRunnerCfg(
        seed=args.seed,
        total_steps=args.total_steps,
        learning_starts=args.learning_starts,
        device=str(wrapped_env.device),
        memory=OffPolicyMemoryCfg(
            capacity=args.buffer_size,
            batch_size=args.batch_size,
        ),
    )
    log_dir = (
        Path(args.log_root)
        / args.exp_name
        / str(cfg.seed)
        / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )

    runner = FastSACRunner(wrapped_env, cfg, log_dir)
    try:
        runner.learn()
    finally:
        wrapped_env.close()


if __name__ == "__main__":
    main()

"""Gymnasium PPO training template."""

import argparse
from datetime import datetime
from pathlib import Path

import gymnasium as gym

from mmrl.env_wrappers import GymnasiumEnvWrapper
from mmrl.ppo import PPOMemoryCfg, PPORunner, PPORunnerCfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("env_id", nargs="?", default="Pendulum-v1")
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--steps-per-env", type=int, default=64)
    parser.add_argument("--device", default=None)
    parser.add_argument("--log-root", default="logs/gymnasium/ppo")
    args = parser.parse_args()

    env = GymnasiumEnvWrapper(gym.make(args.env_id), device=args.device)
    cfg = PPORunnerCfg(
        device=str(env.device),
        max_iterations=args.iterations,
        memory=PPOMemoryCfg(num_steps_per_env=args.steps_per_env),
    )
    log_dir = Path(args.log_root) / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    runner = PPORunner(env, cfg, log_dir)
    try:
        runner.learn()
    finally:
        env.close()


if __name__ == "__main__":
    main()

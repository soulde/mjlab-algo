"""Gymnasium TD-MPC2 training template for dm-control dog-run."""

import argparse
from datetime import datetime
from pathlib import Path

from mmrl.env_wrappers import GymnasiumEnvWrapper
from mmrl.tdmpc2 import TDMPC2Runner, TDMPC2RunnerCfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("env_id", nargs="?", default="dm_control/dog-run-v0")
    parser.add_argument("--steps", type=int, default=1_000_000)
    parser.add_argument("--seed-steps", type=int, default=5_000)
    parser.add_argument("--episode-length", type=int, default=1_000)
    parser.add_argument("--device", default=None)
    parser.add_argument("--log-root", default="logs/gymnasium/tdmpc2")
    args = parser.parse_args()

    env = GymnasiumEnvWrapper.make(args.env_id, device=args.device)
    cfg = TDMPC2RunnerCfg(
        device=str(env.device),
        steps=args.steps,
        seed_steps=args.seed_steps,
        episode_length=args.episode_length,
    )
    log_dir = Path(args.log_root) / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    runner = TDMPC2Runner(env, cfg, log_dir)
    try:
        runner.learn()
    finally:
        env.close()


if __name__ == "__main__":
    main()

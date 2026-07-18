"""Gymnasium TD-MPC2 playback template for dm-control dog-run."""

import argparse

from mmrl.env_wrappers import GymnasiumEnvWrapper
from mmrl.tdmpc2 import TDMPC2Runner, TDMPC2RunnerCfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--env-id", default="dm_control/dog-run-v0")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--episode-length", type=int, default=1_000)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    env = GymnasiumEnvWrapper.make(
        args.env_id, device=args.device, render_mode="human"
    )
    cfg = TDMPC2RunnerCfg(
        device=str(env.device), episode_length=args.episode_length
    )
    runner = TDMPC2Runner(env, cfg, log_dir=".")
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy()
    try:
        for _ in range(args.episodes):
            obs = env.reset()
            done = False
            step = 0
            while not done:
                action = policy(obs, t0=step == 0)
                obs, _reward, done_tensor, _info = env.step(action)
                done = bool(done_tensor.item())
                step += 1
    finally:
        env.close()


if __name__ == "__main__":
    main()

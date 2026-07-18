"""Gymnasium FastSAC playback example."""

import argparse

import gymnasium as gym

from mmrl.env_wrappers.gymnasium import GymnasiumEnvWrapper
from mmrl.fastsac import FastSAC, FastSACRunnerCfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--env-id", default="Pendulum-v1")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = gym.make(args.env_id, render_mode="human")
    wrapped_env = GymnasiumEnvWrapper(env, device=args.device)
    cfg = FastSACRunnerCfg(device=str(wrapped_env.device))
    agent = FastSAC(
        cfg,
        obs_dim=wrapped_env.obs_dim,
        action_dim=wrapped_env.action_dim,
        device=wrapped_env.device,
    )
    agent.load(args.checkpoint)

    try:
        for _episode in range(args.episodes):
            obs = wrapped_env.reset()
            done = False
            while not done:
                action = agent.act(obs, eval_mode=True)
                obs, _reward, done_tensor, _info = wrapped_env.step(action)
                done = bool(done_tensor.item())
    finally:
        wrapped_env.close()


if __name__ == "__main__":
    main()

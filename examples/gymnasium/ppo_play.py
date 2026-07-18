"""Gymnasium PPO playback template."""

import argparse

import gymnasium as gym

from mmrl.env_wrappers import GymnasiumEnvWrapper
from mmrl.ppo import PPORunner, PPORunnerCfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--env-id", default="Pendulum-v1")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    env = GymnasiumEnvWrapper(
        gym.make(args.env_id, render_mode="human"), device=args.device
    )
    runner = PPORunner(env, PPORunnerCfg(device=str(env.device)), log_dir=".")
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy()
    try:
        for _ in range(args.episodes):
            obs = env.reset()
            done = False
            while not done:
                obs, _reward, done_tensor, _info = env.step(policy(obs))
                done = bool(done_tensor.item())
    finally:
        env.close()


if __name__ == "__main__":
    main()

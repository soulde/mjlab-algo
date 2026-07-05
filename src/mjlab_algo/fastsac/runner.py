"""Training runner for FastSAC."""

from pathlib import Path
from time import time

import torch

from mjlab_algo.fastsac.buffer import FastSACReplayBuffer
from mjlab_algo.fastsac.config import FastSACConfig
from mjlab_algo.fastsac.fastsac import FastSAC
from mjlab_algo.fastsac.vecenv_wrapper import FastSACVecEnvWrapper


class FastSACRunner:
    """Online off-policy training loop for FastSAC."""

    def __init__(
        self,
        cfg: FastSACConfig,
        env: FastSACVecEnvWrapper,
        agent: FastSAC,
        buffer: FastSACReplayBuffer,
        log_dir: str | Path,
    ):
        self.cfg = cfg
        self.env = env
        self.agent = agent
        self.buffer = buffer
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._start_time = time()

    def train(self) -> None:
        obs = self.env.reset()
        step = 0
        last_metrics: dict[str, float] = {}
        episode_reward = torch.zeros(self.env.num_envs)

        while step < self.cfg.total_steps:
            if step < self.cfg.learning_starts:
                action = self.env.rand_act()
            else:
                action = self.agent.act(obs)

            next_obs, reward, done, _info = self.env.step(action)
            self.buffer.add(obs, action, reward, next_obs, done)
            episode_reward += reward.cpu()
            obs = next_obs
            step += self.env.num_envs

            if done.any():
                episode_reward[done.cpu()] = 0.0

            should_train = (
                step >= self.cfg.learning_starts
                and self.buffer.size >= self.cfg.batch_size
                and step % self.cfg.train_every == 0
            )
            if should_train:
                for _ in range(self.cfg.gradient_steps):
                    batch = self.buffer.sample(self.cfg.batch_size)
                    last_metrics = self.agent.update(batch)

            if self.cfg.log_interval > 0 and step % self.cfg.log_interval == 0:
                elapsed = max(time() - self._start_time, 1e-6)
                print(
                    f"FastSAC | step: {step:,} | "
                    f"buffer: {self.buffer.size:,} | "
                    f"sps: {step / elapsed:.0f} | "
                    f"alpha: {last_metrics.get('alpha', self.agent.alpha.item()):.3f}"
                )

            if (
                self.cfg.save_agent
                and self.cfg.save_interval > 0
                and step % self.cfg.save_interval == 0
            ):
                self.agent.save(self.log_dir / "models" / f"model_{step}.pt")

        if self.cfg.save_agent:
            self.agent.save(self.log_dir / "models" / "final.pt")

"""Training runner for FastSAC."""

from collections import deque
from pathlib import Path
from time import time

import torch

from mmrl.fastsac.buffer import FastSACReplayBuffer
from mmrl.fastsac.config import FastSACConfig
from mmrl.fastsac.fastsac import FastSAC
from mmrl.env_wrappers.mjlab import MJLabVectorEnvWrapper
from mmrl.logging import format_training_log
from mmrl.runners.off_policy import OffPolicyRunner


class FastSACRunner(OffPolicyRunner):
    """Online off-policy training loop for FastSAC."""

    def __init__(
        self,
        cfg: FastSACConfig,
        env: MJLabVectorEnvWrapper,
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
        self._rewbuffer: deque[float] = deque(maxlen=100)
        self._lenbuffer: deque[float] = deque(maxlen=100)

    def train(self) -> None:
        obs = self.env.reset()
        step = 0
        last_metrics: dict[str, float] = {}
        episode_reward = torch.zeros(self.env.num_envs)
        episode_length = torch.zeros(self.env.num_envs)
        next_log_step = self.cfg.log_interval if self.cfg.log_interval > 0 else -1
        interval_collection_time = 0.0
        interval_learning_time = 0.0
        interval_start_step = 0

        while step < self.cfg.total_steps:
            collect_start = time()
            if step < self.cfg.learning_starts:
                action = self.env.rand_act()
            else:
                action = self.agent.act(obs)

            next_obs, reward, done, _info = self.env.step(action)
            interval_collection_time += time() - collect_start
            self.buffer.add(obs, action, reward, next_obs, done)
            episode_reward += reward.cpu()
            episode_length += 1
            obs = next_obs
            step += self.env.num_envs

            if done.any():
                done_cpu = done.cpu()
                for rew, length in zip(
                    episode_reward[done_cpu],
                    episode_length[done_cpu],
                    strict=False,
                ):
                    self._rewbuffer.append(float(rew))
                    self._lenbuffer.append(float(length))
                episode_reward[done.cpu()] = 0.0
                episode_length[done.cpu()] = 0.0

            should_train = (
                step >= self.cfg.learning_starts
                and self.buffer.size >= self.cfg.batch_size
                and step % self.cfg.train_every == 0
            )
            if should_train:
                learn_start = time()
                for _ in range(self.cfg.gradient_steps):
                    batch = self.buffer.sample(self.cfg.batch_size)
                    last_metrics = self.agent.update(batch)
                interval_learning_time += time() - learn_start

            if self.cfg.log_interval > 0 and step >= next_log_step:
                elapsed = max(time() - self._start_time, 1e-6)
                interval_time = interval_collection_time + interval_learning_time
                interval_steps = max(step - interval_start_step, 1)
                sps = interval_steps / max(interval_time, 1e-6)
                remaining_steps = max(self.cfg.total_steps - step, 0)
                eta = remaining_steps / max(step / elapsed, 1e-6)
                losses = {
                    key.removesuffix("_loss"): value
                    for key, value in last_metrics.items()
                    if key.endswith("_loss")
                }
                mean_reward = (
                    sum(self._rewbuffer) / len(self._rewbuffer)
                    if self._rewbuffer
                    else None
                )
                mean_episode_length = (
                    sum(self._lenbuffer) / len(self._lenbuffer)
                    if self._lenbuffer
                    else None
                )
                print(
                    format_training_log(
                        title=f"FastSAC step {step}/{self.cfg.total_steps}",
                        total_steps=step,
                        steps_per_second=sps,
                        collection_time=interval_collection_time,
                        learning_time=interval_learning_time,
                        losses=losses,
                        mean_reward=mean_reward,
                        mean_episode_length=mean_episode_length,
                        extras={
                            "Replay buffer": self.buffer.size,
                            "Alpha": last_metrics.get("alpha", self.agent.alpha.item()),
                        },
                        iteration_time=interval_time,
                        elapsed_time=elapsed,
                        eta_seconds=eta,
                        log_dir=self.log_dir,
                    )
                )
                interval_collection_time = 0.0
                interval_learning_time = 0.0
                interval_start_step = step
                while next_log_step <= step:
                    next_log_step += self.cfg.log_interval

            if (
                self.cfg.save_agent
                and self.cfg.save_interval > 0
                and step % self.cfg.save_interval == 0
            ):
                self.agent.save(self.log_dir / "models" / f"model_{step}.pt")

        if self.cfg.save_agent:
            self.agent.save(self.log_dir / "models" / "final.pt")

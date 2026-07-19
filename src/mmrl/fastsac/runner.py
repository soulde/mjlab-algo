"""Training runner for FastSAC."""

from collections import deque
from pathlib import Path
from time import time

import torch

from mmrl.config import config_to_dict, get_config_value
from mmrl.fastsac.fastsac import FastSAC
from mmrl.env_wrappers import EnvWrapper
from mmrl.logging import MetricLogger, format_training_log
from mmrl.memories import OffPolicyReplayMemory
from mmrl.runners.off_policy import OffPolicyRunner


class FastSACRunner(OffPolicyRunner):
    """Online off-policy training loop for FastSAC."""

    def __init__(
        self,
        env: EnvWrapper,
        train_cfg,
        log_dir: str | Path,
        device: str | torch.device | None = None,
    ):
        self.cfg = train_cfg
        self.env = env
        self.device = torch.device(
            device or get_config_value(train_cfg, "device") or env.device
        )
        self._validate_components()
        self.agent = FastSAC(
            train_cfg,
            obs_dim=env.obs_dim,
            action_dim=env.action_dim,
            device=self.device,
        )
        self.buffer = OffPolicyReplayMemory(
            capacity=get_config_value(train_cfg, "memory.capacity"),
            obs_dim=env.obs_dim,
            action_dim=env.action_dim,
            device=self.device,
        )
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.logger = MetricLogger(
            self.log_dir,
            get_config_value(train_cfg, "logger"),
            config_to_dict(train_cfg),
        )
        self._start_time = time()
        self._rewbuffer: deque[float] = deque(maxlen=100)
        self._lenbuffer: deque[float] = deque(maxlen=100)

    def _validate_components(self) -> None:
        supported = {
            "algorithm.class_name": "FastSAC",
            "actor.class_name": "SquashedGaussianActor",
            "critic.class_name": "TwinQNetwork",
            "memory.class_name": "OffPolicyReplayMemory",
        }
        for path, expected in supported.items():
            actual = get_config_value(self.cfg, path)
            if actual != expected:
                raise ValueError(
                    f"Unsupported {path} {actual!r}; FastSACRunner supports {expected!r}."
                )

    def get_inference_policy(self, device: str | torch.device | None = None):
        """Return a deterministic policy callable for environment play scripts."""
        if device is not None:
            self.agent.device = torch.device(device)
            self.agent.actor.to(self.agent.device)

        def policy(obs: torch.Tensor) -> torch.Tensor:
            return self.agent.act(obs, eval_mode=True)

        return policy

    def save(self, path: str | Path) -> None:
        self.agent.save(path)

    def load(self, path: str | Path) -> None:
        self.agent.load(path)

    def learn(self) -> None:
        obs = self.env.reset()
        step = 0
        last_metrics: dict[str, float] = {}
        episode_reward = torch.zeros(self.env.num_envs)
        episode_length = torch.zeros(self.env.num_envs)
        log_interval = get_config_value(self.cfg, "log_interval")
        total_steps = get_config_value(self.cfg, "total_steps")
        next_log_step = log_interval if log_interval > 0 else -1
        interval_collection_time = 0.0
        interval_learning_time = 0.0
        interval_start_step = 0

        while step < total_steps:
            collect_start = time()
            if step < get_config_value(self.cfg, "learning_starts"):
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
                step >= get_config_value(self.cfg, "learning_starts")
                and self.buffer.size >= get_config_value(self.cfg, "memory.batch_size")
                and step % get_config_value(self.cfg, "train_every") == 0
            )
            if should_train:
                learn_start = time()
                for _ in range(get_config_value(self.cfg, "gradient_steps")):
                    batch = self.buffer.sample(
                        get_config_value(self.cfg, "memory.batch_size")
                    )
                    last_metrics = self.agent.update(batch)
                interval_learning_time += time() - learn_start

            if log_interval > 0 and step >= next_log_step:
                elapsed = max(time() - self._start_time, 1e-6)
                interval_time = interval_collection_time + interval_learning_time
                interval_steps = max(step - interval_start_step, 1)
                sps = interval_steps / max(interval_time, 1e-6)
                remaining_steps = max(total_steps - step, 0)
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
                        title=f"FastSAC step {step}/{total_steps}",
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
                        color=get_config_value(self.cfg, "logger.color", True),
                    )
                )
                metric_values = {
                    **last_metrics,
                    "replay_size": self.buffer.size,
                    "steps_per_second": sps,
                    "collection_time": interval_collection_time,
                    "learning_time": interval_learning_time,
                }
                if mean_reward is not None:
                    metric_values["mean_episode_reward"] = mean_reward
                if mean_episode_length is not None:
                    metric_values["mean_episode_length"] = mean_episode_length
                self.logger.log(metric_values, step=step, prefix="train")
                interval_collection_time = 0.0
                interval_learning_time = 0.0
                interval_start_step = step
                while next_log_step <= step:
                    next_log_step += log_interval

            if (
                get_config_value(self.cfg, "save_interval") > 0
                and step % get_config_value(self.cfg, "save_interval") == 0
            ):
                self.save(self.log_dir / "models" / f"model_{step}.pt")

        self.save(self.log_dir / "models" / "final.pt")
        self.logger.close()

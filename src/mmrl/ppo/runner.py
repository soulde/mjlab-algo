"""On-policy runner for PPO."""

from collections import deque
from pathlib import Path
from time import time

import torch

from mmrl.config import config_to_dict, get_config_value
from mmrl.env_wrappers import EnvWrapper
from mmrl.logging import format_training_log
from mmrl.memories import OnPolicyRolloutMemory
from mmrl.ppo.actor_critic import ActorCritic
from mmrl.ppo.ppo import PPO
from mmrl.runners import OnPolicyRunner


class PPORunner(OnPolicyRunner):
    """Collect vectorized rollouts and optimize an mmrl PPO policy."""

    def __init__(
        self,
        env: EnvWrapper,
        train_cfg,
        log_dir: str | Path,
        device: str | torch.device | None = None,
    ):
        self.env = env
        self.cfg = train_cfg
        self.device = torch.device(
            device or get_config_value(train_cfg, "device") or env.device
        )
        self._validate_components()
        self.policy = ActorCritic(
            get_config_value(train_cfg, "actor_critic"),
            obs_dim=env.obs_dim,
            action_dim=env.action_dim,
        ).to(self.device)
        self.algorithm = PPO(
            get_config_value(train_cfg, "algorithm"),
            self.policy,
            self.device,
        )
        self.memory = OnPolicyRolloutMemory(
            num_steps=get_config_value(train_cfg, "memory.num_steps_per_env"),
            num_envs=env.num_envs,
            obs_shape=(env.obs_dim,),
            action_shape=(env.action_dim,),
            device=self.device,
        )
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.current_iteration = 0
        self._episode_rewards = torch.zeros(env.num_envs)
        self._episode_lengths = torch.zeros(env.num_envs)
        self._reward_history: deque[float] = deque(maxlen=100)
        self._length_history: deque[float] = deque(maxlen=100)

    def _validate_components(self) -> None:
        supported = {
            "actor_critic.class_name": "ActorCritic",
            "algorithm.class_name": "PPO",
            "memory.class_name": "OnPolicyRolloutMemory",
        }
        for path, expected in supported.items():
            actual = get_config_value(self.cfg, path)
            if actual != expected:
                raise ValueError(
                    f"Unsupported {path} {actual!r}; PPORunner supports {expected!r}."
                )

    def learn(self) -> None:
        obs = self.env.reset().to(self.device)
        start_time = time()
        max_iterations = get_config_value(self.cfg, "max_iterations")
        log_interval = get_config_value(self.cfg, "log_interval")
        for iteration in range(self.current_iteration, max_iterations):
            collection_start = time()
            obs = self._collect_rollout(obs)
            collection_time = time() - collection_start

            with torch.no_grad():
                last_value = self.policy.critic(obs)
            algorithm_cfg = get_config_value(self.cfg, "algorithm")
            self.memory.compute_returns(
                last_value,
                gamma=algorithm_cfg.gamma,
                gae_lambda=algorithm_cfg.lam,
                normalize_advantage=(
                    not algorithm_cfg.normalize_advantage_per_mini_batch
                ),
            )
            learning_start = time()
            metrics = self.algorithm.update(self.memory)
            learning_time = time() - learning_start
            self.memory.clear()
            self.current_iteration = iteration + 1

            if log_interval > 0 and self.current_iteration % log_interval == 0:
                self._log(metrics, collection_time, learning_time, start_time)
            save_interval = get_config_value(self.cfg, "save_interval")
            if save_interval > 0 and self.current_iteration % save_interval == 0:
                self.save(
                    self.log_dir / "models" / f"model_{self.current_iteration}.pt"
                )
        self.save(self.log_dir / "models" / "final.pt")

    @torch.no_grad()
    def _collect_rollout(self, obs: torch.Tensor) -> torch.Tensor:
        num_steps = get_config_value(self.cfg, "memory.num_steps_per_env")
        gamma = get_config_value(self.cfg, "algorithm.gamma")
        for _ in range(num_steps):
            action, log_prob, value = self.algorithm.act(obs)
            next_obs, reward, done, info = self.env.step(action)
            reward = reward.to(self.device, dtype=torch.float32)
            done = done.to(self.device, dtype=torch.bool)
            time_outs = info.get("time_outs") if isinstance(info, dict) else None
            if time_outs is not None:
                reward = reward + gamma * value.squeeze(-1) * torch.as_tensor(
                    time_outs, device=self.device, dtype=torch.float32
                )
            self.memory.add(obs, action, reward, done, log_prob, value)
            self._track_episodes(reward, done)
            obs = next_obs.to(self.device)
        return obs

    def _track_episodes(self, reward: torch.Tensor, done: torch.Tensor) -> None:
        self._episode_rewards += reward.detach().cpu()
        self._episode_lengths += 1
        done_cpu = done.detach().cpu()
        for episode_reward, episode_length in zip(
            self._episode_rewards[done_cpu],
            self._episode_lengths[done_cpu],
            strict=False,
        ):
            self._reward_history.append(float(episode_reward))
            self._length_history.append(float(episode_length))
        self._episode_rewards[done_cpu] = 0
        self._episode_lengths[done_cpu] = 0

    def _log(
        self,
        metrics: dict[str, float],
        collection_time: float,
        learning_time: float,
        start_time: float,
    ) -> None:
        elapsed = time() - start_time
        steps = (
            self.current_iteration
            * self.memory.storage.num_steps
            * self.env.num_envs
        )
        iteration_time = collection_time + learning_time
        print(
            format_training_log(
                title=(
                    f"PPO iteration {self.current_iteration}/"
                    f"{get_config_value(self.cfg, 'max_iterations')}"
                ),
                total_steps=steps,
                steps_per_second=(
                    self.memory.storage.num_steps
                    * self.env.num_envs
                    / max(iteration_time, 1e-6)
                ),
                collection_time=collection_time,
                learning_time=learning_time,
                losses={
                    "value": metrics["value_loss"],
                    "surrogate": metrics["surrogate_loss"],
                },
                mean_reward=(
                    sum(self._reward_history) / len(self._reward_history)
                    if self._reward_history
                    else None
                ),
                mean_episode_length=(
                    sum(self._length_history) / len(self._length_history)
                    if self._length_history
                    else None
                ),
                extras={
                    "Entropy": metrics["entropy"],
                    "KL": metrics["kl"],
                    "Learning rate": metrics["learning_rate"],
                },
                iteration_time=iteration_time,
                elapsed_time=elapsed,
                eta_seconds=(
                    elapsed
                    / max(self.current_iteration, 1)
                    * (
                        get_config_value(self.cfg, "max_iterations")
                        - self.current_iteration
                    )
                ),
                log_dir=self.log_dir,
            )
        )

    def get_inference_policy(self, device: str | torch.device | None = None):
        if device is not None:
            self.device = torch.device(device)
            self.policy.to(self.device)

        def policy(obs: torch.Tensor) -> torch.Tensor:
            with torch.inference_mode():
                return self.policy.act_inference(obs.to(self.device))

        return policy

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "policy": self.policy.state_dict(),
                "optimizer": self.algorithm.optimizer.state_dict(),
                "iteration": self.current_iteration,
                "cfg": config_to_dict(self.cfg),
            },
            path,
        )

    def load(self, path: str | Path) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.policy.load_state_dict(checkpoint["policy"])
        self.algorithm.optimizer.load_state_dict(checkpoint["optimizer"])
        self.current_iteration = int(checkpoint.get("iteration", 0))

"""On-policy training runner for Adversarial Motion Priors."""

from collections import deque
from pathlib import Path
from time import time

import torch

from mmrl.amp.amp import AMP
from mmrl.config import config_to_dict, get_config_value
from mmrl.env_wrappers import EnvWrapper
from mmrl.logging import MetricLogger, format_training_log
from mmrl.memories import AMPExpertSource, OnPolicyRolloutMemory
from mmrl.models import AMPDiscriminator
from mmrl.ppo import ActorCritic
from mmrl.runners import OnPolicyRunner


class AMPRunner(OnPolicyRunner):
    """Collect PPO rollouts and train an AMP motion discriminator."""

    def __init__(
        self,
        env: EnvWrapper,
        train_cfg,
        expert_source: AMPExpertSource,
        log_dir: str | Path,
        device: str | torch.device | None = None,
    ) -> None:
        self.env = env
        self.cfg = train_cfg
        self.device = torch.device(
            device or get_config_value(train_cfg, "device") or env.device
        )
        self._validate_components()
        amp_observation = self._get_amp_observations()
        amp_observation_dim = int(amp_observation.shape[-1])
        if expert_source.observation_dim != amp_observation_dim:
            raise ValueError(
                "Expert AMP observation dimension "
                f"{expert_source.observation_dim} does not match environment "
                f"dimension {amp_observation_dim}."
            )
        rollout_size = (
            get_config_value(train_cfg, "memory.num_steps_per_env") * env.num_envs
        )
        num_mini_batches = get_config_value(
            train_cfg, "algorithm.num_mini_batches"
        )
        if rollout_size % num_mini_batches != 0:
            raise ValueError(
                f"Rollout size {rollout_size} must be divisible by "
                f"num_mini_batches {num_mini_batches}."
            )
        self.policy = ActorCritic(
            get_config_value(train_cfg, "actor_critic"),
            obs_dim=env.obs_dim,
            action_dim=env.action_dim,
        ).to(self.device)
        discriminator_cfg = get_config_value(train_cfg, "discriminator")
        self.discriminator = AMPDiscriminator(
            amp_observation_dim,
            tuple(get_config_value(discriminator_cfg, "hidden_dims")),
        ).to(self.device)
        self.algorithm = AMP(
            get_config_value(train_cfg, "algorithm"),
            self.policy,
            self.discriminator,
            expert_source,
            amp_observation_dim,
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
        self.logger = MetricLogger(
            self.log_dir,
            get_config_value(train_cfg, "logger"),
            config_to_dict(train_cfg),
        )
        self.current_iteration = 0
        self._episode_rewards = torch.zeros(env.num_envs)
        self._episode_lengths = torch.zeros(env.num_envs)
        self._reward_history: deque[float] = deque(maxlen=100)
        self._length_history: deque[float] = deque(maxlen=100)

    def _validate_components(self) -> None:
        supported = {
            "actor_critic.class_name": "ActorCritic",
            "algorithm.class_name": "AMP",
            "discriminator.class_name": "AMPDiscriminator",
            "memory.class_name": "OnPolicyRolloutMemory",
        }
        for path, expected in supported.items():
            actual = get_config_value(self.cfg, path)
            if actual != expected:
                raise ValueError(
                    f"Unsupported {path} {actual!r}; AMPRunner supports "
                    f"{expected!r}."
                )

    def _get_amp_observations(self) -> torch.Tensor:
        return self.env.get_amp_observations().to(self.device)

    def learn(self) -> None:
        obs = self.env.reset().to(self.device)
        amp_obs = self._get_amp_observations()
        start_time = time()
        max_iterations = get_config_value(self.cfg, "max_iterations")
        log_interval = get_config_value(self.cfg, "log_interval")
        for iteration in range(self.current_iteration, max_iterations):
            collection_start = time()
            obs, amp_obs = self._collect_rollout(obs, amp_obs)
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
        self.logger.close()

    @torch.no_grad()
    def _collect_rollout(
        self, obs: torch.Tensor, amp_obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_steps = get_config_value(self.cfg, "memory.num_steps_per_env")
        algorithm_cfg = get_config_value(self.cfg, "algorithm")
        discriminator_cfg = get_config_value(self.cfg, "discriminator")
        for _ in range(num_steps):
            action, log_prob, value = self.algorithm.act(obs)
            next_obs, task_reward, done, info = self.env.step(action)
            next_obs = next_obs.to(self.device)
            done = done.to(self.device, dtype=torch.bool)
            next_amp_obs = self._get_amp_observations()
            replay_next_amp_obs = self._terminal_amp_observations(
                next_amp_obs, done, info
            )
            reward, _style_reward = self.algorithm.combine_rewards(
                amp_obs,
                replay_next_amp_obs,
                task_reward,
                discriminator_cfg.reward_scale,
                discriminator_cfg.task_reward_lerp,
            )
            time_outs = info.get("time_outs") if isinstance(info, dict) else None
            if time_outs is not None:
                reward += (
                    algorithm_cfg.gamma
                    * value.squeeze(-1)
                    * torch.as_tensor(
                        time_outs, device=self.device, dtype=torch.float32
                    )
                )
            self.algorithm.add_amp_transition(amp_obs, replay_next_amp_obs)
            self.memory.add(obs, action, reward, done, log_prob, value)
            self._track_episodes(reward, done)
            obs = next_obs
            amp_obs = next_amp_obs
        return obs, amp_obs

    def _terminal_amp_observations(
        self, next_amp_obs: torch.Tensor, done: torch.Tensor, info
    ) -> torch.Tensor:
        if not isinstance(info, dict) or "terminal_amp_observations" not in info:
            return next_amp_obs
        result = next_amp_obs.clone()
        terminal = torch.as_tensor(
            info["terminal_amp_observations"],
            device=self.device,
            dtype=torch.float32,
        )
        done_ids = done.nonzero(as_tuple=False).flatten()
        if terminal.shape[0] == self.env.num_envs:
            result[done_ids] = terminal[done_ids]
        elif terminal.shape[0] == done_ids.numel():
            result[done_ids] = terminal
        else:
            raise ValueError(
                "terminal_amp_observations must contain all environments or "
                "exactly the completed environments."
            )
        return result

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
        steps_per_second = (
            self.memory.storage.num_steps
            * self.env.num_envs
            / max(iteration_time, 1e-6)
        )
        mean_reward = (
            sum(self._reward_history) / len(self._reward_history)
            if self._reward_history
            else None
        )
        mean_length = (
            sum(self._length_history) / len(self._length_history)
            if self._length_history
            else None
        )
        print(
            format_training_log(
                title=(
                    f"AMP iteration {self.current_iteration}/"
                    f"{get_config_value(self.cfg, 'max_iterations')}"
                ),
                total_steps=steps,
                steps_per_second=steps_per_second,
                collection_time=collection_time,
                learning_time=learning_time,
                losses={
                    "value": metrics["value_loss"],
                    "surrogate": metrics["surrogate_loss"],
                    "AMP": metrics["amp_loss"],
                    "AMP gradient penalty": metrics[
                        "amp_gradient_penalty"
                    ],
                },
                mean_reward=mean_reward,
                mean_episode_length=mean_length,
                extras={
                    "AMP policy prediction": metrics[
                        "amp_policy_prediction"
                    ],
                    "AMP expert prediction": metrics[
                        "amp_expert_prediction"
                    ],
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
                color=get_config_value(self.cfg, "logger.color", True),
            )
        )
        metric_values = {
            **metrics,
            "steps_per_second": steps_per_second,
            "collection_time": collection_time,
            "learning_time": learning_time,
        }
        if mean_reward is not None:
            metric_values["mean_episode_reward"] = mean_reward
        if mean_length is not None:
            metric_values["mean_episode_length"] = mean_length
        self.logger.log(metric_values, step=steps, prefix="train")

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
                "discriminator": self.discriminator.state_dict(),
                "normalizer": self.algorithm.normalizer.state_dict(),
                "optimizer": self.algorithm.optimizer.state_dict(),
                "iteration": self.current_iteration,
                "cfg": config_to_dict(self.cfg),
            },
            path,
        )

    def load(self, path: str | Path) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.policy.load_state_dict(checkpoint["policy"])
        self.discriminator.load_state_dict(checkpoint["discriminator"])
        self.algorithm.normalizer.load_state_dict(checkpoint["normalizer"])
        self.algorithm.optimizer.load_state_dict(checkpoint["optimizer"])
        self.current_iteration = int(checkpoint.get("iteration", 0))

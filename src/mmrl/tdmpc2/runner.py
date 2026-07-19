"""TD-MPC2 training runner.

Encapsulates the online training loop: environment interaction,
experience collection, agent updates, evaluation, and logging.
"""

from __future__ import annotations

from pathlib import Path
from time import time
import numpy as np
import torch

from mmrl.config import config_to_dict, get_config_value
from mmrl.logging import MetricLogger, format_training_log
from mmrl.runners.model_based import ModelBasedRunner
from mmrl.env_wrappers import EnvWrapper
from mmrl.memories import EpisodeMemory

from mmrl.models import WorldModel
from mmrl.tdmpc2.config import TDMPC2EnvSpec
from mmrl.tdmpc2.tdmpc2 import TDMPC2


class TDMPC2Runner(ModelBasedRunner):
    """Runner for vectorized online single-task TD-MPC2 training.

    Manages the training loop: collects experience, updates
    the agent, evaluates periodically, and logs metrics.
    """

    def __init__(
        self,
        env: EnvWrapper,
        train_cfg,
        log_dir: Path,
        device: str | torch.device | None = None,
    ):
        algorithm_name = get_config_value(train_cfg, "algorithm.class_name")
        model_name = get_config_value(train_cfg, "model.class_name")
        memory_name = get_config_value(train_cfg, "memory.class_name")
        if algorithm_name != "TDMPC2":
            raise ValueError(f"Unsupported algorithm.class_name {algorithm_name!r}.")
        if model_name != "WorldModel":
            raise ValueError(f"Unsupported model.class_name {model_name!r}.")
        if memory_name != "EpisodeMemory":
            raise ValueError(
                f"Unsupported memory.class_name {memory_name!r}."
            )
        self.train_cfg = train_cfg
        self.cfg = train_cfg
        self.env = env
        self.device = torch.device(
            device or get_config_value(train_cfg, "device") or env.device
        )
        episode_length = get_config_value(train_cfg, "episode_length", 0)
        if episode_length <= 0:
            episode_length = int(getattr(env.unwrapped, "max_episode_length", 0))
        if episode_length <= 0:
            raise ValueError(
                "TDMPC2RunnerCfg.episode_length must be set when the environment "
                "does not expose max_episode_length."
            )
        self.env_spec = TDMPC2EnvSpec(
            obs_shape={"state": (env.obs_dim,)},
            action_dim=env.action_dim,
            episode_length=episode_length,
        )
        self.seed_steps = get_config_value(train_cfg, "seed_steps")
        if self.seed_steps <= 0:
            self.seed_steps = max(1000, 5 * episode_length)
        algorithm_cfg = get_config_value(train_cfg, "algorithm")
        model_cfg = get_config_value(train_cfg, "model")
        self.model = WorldModel(model_cfg, algorithm_cfg, self.env_spec)
        self.agent = TDMPC2(
            algorithm_cfg,
            self.model,
            self.env_spec,
            batch_size=get_config_value(train_cfg, "memory.batch_size"),
            device=self.device,
        )
        self.buffer = EpisodeMemory(
            capacity=get_config_value(train_cfg, "memory.capacity"),
            batch_size=get_config_value(train_cfg, "memory.batch_size"),
            horizon=get_config_value(train_cfg, "algorithm.horizon"),
            device=self.device,
        )
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.logger = MetricLogger(
            self.log_dir,
            get_config_value(train_cfg, "logger"),
            config_to_dict(train_cfg),
        )

        self._step = 0
        self._ep_idx = 0
        self._start_time = time()

    def get_inference_policy(self, device: str | torch.device | None = None):
        """Return the TD-MPC2 policy callable used by play scripts."""
        if device is not None:
            self.agent.device = torch.device(device)
            self.agent.model.to(self.agent.device)

        def policy(obs: torch.Tensor, t0: bool = False) -> torch.Tensor:
            return self.agent.act(obs, t0=t0, eval_mode=True)

        return policy

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.agent.save(str(path))

    def load(self, path: str | Path) -> None:
        self.agent.load(path)

    def _common_metrics(self) -> dict:
        """Return dictionary of common metrics (step, episode, SPS)."""
        elapsed = time() - self._start_time
        return dict(
            step=self._step,
            episode=self._ep_idx,
            elapsed_time=elapsed,
            steps_per_second=self._step / elapsed if elapsed > 0 else 0,
        )

    def _log(self, metrics: dict, category: str) -> None:
        """Log metrics to the console and configured metric backends."""
        step = int(metrics.get("step", self._step))
        self.logger.log(metrics, step=step, prefix=category)
        elapsed = float(metrics.get("elapsed_time", time() - self._start_time))
        eta = (
            (self.cfg.steps - step) / max(step / max(elapsed, 1e-6), 1e-6)
            if step > 0
            else 0.0
        )
        title = (
            f"TD-MPC2 eval step {step}/{self.cfg.steps}"
            if category == "eval"
            else f"TD-MPC2 step {step}/{self.cfg.steps}"
        )
        losses = {
            key.removesuffix("_loss"): value
            for key, value in metrics.items()
            if key.endswith("_loss")
        }
        extras = {
            "Episode": metrics.get("episode", self._ep_idx),
            "Buffer episodes": self.buffer.num_eps,
            "Success": metrics.get("episode_success", 0.0),
        }
        if "alpha" in metrics:
            extras["Alpha"] = metrics["alpha"]
        print(
            format_training_log(
                title=title,
                total_steps=step,
                steps_per_second=float(metrics.get("steps_per_second", 0.0)),
                collection_time=float(metrics.get("collection_time", 0.0)),
                learning_time=float(metrics.get("learning_time", 0.0)),
                losses=losses,
                mean_reward=float(metrics["episode_reward"])
                if "episode_reward" in metrics
                else None,
                mean_episode_length=float(metrics["episode_length"])
                if "episode_length" in metrics
                else None,
                extras=extras,
                iteration_time=float(metrics.get("iteration_time", 0.0)),
                elapsed_time=elapsed,
                eta_seconds=eta,
                log_dir=self.log_dir,
                color=get_config_value(self.cfg, "logger.color", True),
            )
        )

    def _to_td(
        self,
        obs: torch.Tensor,
        action: torch.Tensor | None = None,
        reward: torch.Tensor | None = None,
        terminated: torch.Tensor | None = None,
    ):
        """Create a TensorDict for one environment timestep."""
        from tensordict import TensorDict

        if action is None:
            action = torch.full(
                (self.env.action_dim,), float("nan"), device=obs.device
            )
        action = action.to(obs.device)
        if reward is None:
            reward = torch.tensor(float("nan"), device=obs.device)
        else:
            reward = reward.to(obs.device).reshape(())
        if terminated is None:
            terminated = torch.tensor(float("nan"), device=obs.device)
        elif isinstance(terminated, torch.Tensor):
            terminated = terminated.to(obs.device).reshape(())
        else:
            terminated = torch.tensor(float(terminated), device=obs.device)
        td = TensorDict(
            obs=obs.unsqueeze(0),
            action=action.unsqueeze(0),
            reward=reward.unsqueeze(0),
            terminated=terminated.unsqueeze(0),
            batch_size=(1,),
        )
        return td

    def _reset_env(self) -> torch.Tensor:
        obs = self.env.reset()
        if obs.shape != (self.env.num_envs, self.env.obs_dim):
            raise ValueError(
                "Vector environment reset observations must have shape "
                f"({self.env.num_envs}, {self.env.obs_dim}), got {tuple(obs.shape)}."
            )
        return obs

    def _step_env(
        self, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        obs, reward, done, info = self.env.step(action)
        return obs, reward.view(-1), done.view(-1).bool(), info

    def _random_action(self) -> torch.Tensor:
        return self.env.rand_act()

    @staticmethod
    def _info_at(info: dict, key: str, index: int, default=0.0):
        value = info.get(key, default)
        if isinstance(value, torch.Tensor):
            return value.reshape(-1)[index]
        if isinstance(value, (list, tuple, np.ndarray)):
            return value[index]
        return value

    def eval(self) -> dict:
        """Evaluate agent for ``eval_episodes`` episodes."""
        ep_rewards: list[float] = []
        ep_successes: list[float] = []
        ep_lengths: list[int] = []
        if self.cfg.eval_episodes <= 0:
            return {}

        obs = self._reset_env()
        rewards = torch.zeros(self.env.num_envs, device=self.device)
        lengths = torch.zeros(self.env.num_envs, dtype=torch.long, device=self.device)
        t0 = torch.ones(self.env.num_envs, dtype=torch.bool, device=self.device)
        while len(ep_rewards) < self.cfg.eval_episodes:
            action = self.agent.act(obs, t0=t0, eval_mode=True)
            obs, reward, done, info = self._step_env(action)
            rewards += reward.to(self.device)
            lengths += 1
            for index in done.nonzero(as_tuple=False).flatten().tolist():
                if len(ep_rewards) >= self.cfg.eval_episodes:
                    break
                ep_rewards.append(float(rewards[index]))
                ep_successes.append(
                    float(self._info_at(info, "success", index, 0.0))
                )
                ep_lengths.append(int(lengths[index]))
                rewards[index] = 0
                lengths[index] = 0
            t0 = done.to(self.device)

        return dict(
            episode_reward=np.nanmean(ep_rewards),
            episode_success=np.nanmean(ep_successes),
            episode_length=np.nanmean(ep_lengths),
        )

    def learn(self) -> None:
        """Run the main TD-MPC2 training loop."""
        train_metrics: dict = {}
        next_log_step = self.cfg.log_freq if self.cfg.log_freq > 0 else -1
        next_eval_step = 0
        pretrained = False

        print(f"Training on device: {self.agent.device}")
        print(f"Log directory: {self.log_dir}")

        obs = self._reset_env()
        trajectories = [[self._to_td(obs[index])] for index in range(self.env.num_envs)]
        episode_rewards = torch.zeros(self.env.num_envs, device=self.device)
        episode_lengths = torch.zeros(
            self.env.num_envs, dtype=torch.long, device=self.device
        )
        t0 = torch.ones(self.env.num_envs, dtype=torch.bool, device=self.device)

        while self._step < self.cfg.steps:
            if self.cfg.eval_freq > 0 and self._step >= next_eval_step:
                eval_metrics = self.eval()
                eval_metrics.update(self._common_metrics())
                self._log(eval_metrics, "eval")
                while next_eval_step <= self._step:
                    next_eval_step += self.cfg.eval_freq
                obs = self._reset_env()
                trajectories = [
                    [self._to_td(obs[index])] for index in range(self.env.num_envs)
                ]
                episode_rewards.zero_()
                episode_lengths.zero_()
                t0.fill_(True)

            collect_start = time()
            if self._step > self.seed_steps:
                action = self.agent.act(obs, t0=t0)
            else:
                action = self._random_action()
            obs, reward, done, info = self._step_env(action)
            terminated = info.get("terminated", done)
            terminated = torch.as_tensor(terminated, device=self.device).view(-1)
            episode_rewards += reward.to(self.device)
            episode_lengths += 1
            for index in range(self.env.num_envs):
                trajectories[index].append(
                    self._to_td(
                        obs[index], action[index], reward[index], terminated[index]
                    )
                )
            completed_rewards = []
            completed_lengths = []
            completed_successes = []
            for index in done.nonzero(as_tuple=False).flatten().tolist():
                if bool(terminated[index]) and not get_config_value(
                    self.cfg, "algorithm.episodic"
                ):
                    raise ValueError(
                        "Termination detected but episodic mode is off. "
                        "Set ``episodic=True`` to enable terminations."
                    )
                self._ep_idx = self.buffer.add(torch.cat(trajectories[index]))
                completed_rewards.append(float(episode_rewards[index]))
                completed_lengths.append(int(episode_lengths[index]))
                completed_successes.append(
                    float(self._info_at(info, "success", index, 0.0))
                )
                trajectories[index] = [self._to_td(obs[index])]
                episode_rewards[index] = 0
                episode_lengths[index] = 0
            collect_time = time() - collect_start

            # Update agent
            learn_time = 0.0
            collected_steps = self._step + self.env.num_envs
            if collected_steps >= self.seed_steps and self.buffer.num_eps > 0:
                if not pretrained:
                    num_updates = self.seed_steps
                    print("Pretraining agent on seed data...")
                    pretrained = True
                else:
                    num_updates = 1
                learn_start = time()
                _train_metrics = {}
                for _ in range(num_updates):
                    _train_metrics = self.agent.update(self.buffer)
                train_metrics.update(_train_metrics)
                learn_time = time() - learn_start
            train_metrics["collection_time"] = collect_time
            train_metrics["learning_time"] = learn_time
            train_metrics["iteration_time"] = collect_time + learn_time

            self._step = collected_steps
            t0 = done.to(self.device)
            if completed_rewards:
                train_metrics.update(
                    episode_reward=float(np.mean(completed_rewards)),
                    episode_success=float(np.mean(completed_successes)),
                    episode_length=float(np.mean(completed_lengths)),
                )
            if self.cfg.log_freq > 0 and self._step >= next_log_step:
                log_metrics = dict(train_metrics)
                log_metrics.update(self._common_metrics())
                self._log(log_metrics, "train")
                while next_log_step <= self._step:
                    next_log_step += self.cfg.log_freq

        self._finish()

    def _finish(self) -> None:
        """Save final checkpoint and clean up."""
        # Save final agent
        if self.cfg.save_agent:
            save_path = self.log_dir / "models" / "final.pt"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            self.save(save_path)
            print(f"Saved final model to {save_path}")

        self.logger.close()

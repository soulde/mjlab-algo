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
    """Runner for online single-task TD-MPC2 training.

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
        if env.num_envs != 1:
            raise ValueError("TDMPC2Runner requires a single environment.")
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
            if obs.ndim > 1 and obs.shape[0] == 1:
                obs = obs.squeeze(0)
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
            )
        )

    def _to_td(
        self,
        obs: torch.Tensor,
        action: torch.Tensor | None = None,
        reward: torch.Tensor | None = None,
        terminated: torch.Tensor | None = None,
    ):
        """Create a TensorDict for a single timestep."""
        from tensordict import TensorDict

        if action is None:
            action = torch.full_like(self._random_action(), float("nan"))
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
        if obs.ndim > 1 and obs.shape[0] == 1:
            obs = obs.squeeze(0)
        return obs

    def _step_env(
        self, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, bool, dict]:
        obs, reward, done, info = self.env.step(action)
        if obs.ndim > 1 and obs.shape[0] == 1:
            obs = obs.squeeze(0)
        reward = reward.reshape(())
        if isinstance(done, torch.Tensor):
            done = bool(done.reshape(()).item())
        return obs, reward, done, info

    def _random_action(self) -> torch.Tensor:
        action = self.env.rand_act()
        if action.ndim > 1 and action.shape[0] == 1:
            action = action.squeeze(0)
        return action

    def eval(self) -> dict:
        """Evaluate agent for ``eval_episodes`` episodes."""
        ep_rewards: list[float] = []
        ep_successes: list[float] = []
        ep_lengths: list[int] = []
        if self.cfg.eval_episodes <= 0:
            return {}

        for _i in range(self.cfg.eval_episodes):
            obs = self._reset_env()
            done = False
            info: dict = {}
            ep_reward = 0.0
            t = 0
            while not done:
                action = self.agent.act(obs, t0=t == 0, eval_mode=True)
                obs, reward, done, info = self._step_env(action)
                ep_reward += float(reward.item())
                t += 1
            ep_rewards.append(ep_reward)
            ep_successes.append(float(info.get("success", 0.0)))
            ep_lengths.append(t)

        return dict(
            episode_reward=np.nanmean(ep_rewards),
            episode_success=np.nanmean(ep_successes),
            episode_length=np.nanmean(ep_lengths),
        )

    def learn(self) -> None:
        """Run the main TD-MPC2 training loop."""
        train_metrics: dict = {}
        done = True
        eval_next = False
        info: dict = {}
        next_log_step = self.cfg.log_freq if self.cfg.log_freq > 0 else -1

        print(f"Training on device: {self.agent.device}")
        print(f"Log directory: {self.log_dir}")

        while self._step < self.cfg.steps:
            # Evaluate periodically
            if self._step % self.cfg.eval_freq == 0:
                eval_next = True

            # Reset environment
            if done:
                if eval_next:
                    eval_metrics = self.eval()
                    eval_metrics.update(self._common_metrics())
                    self._log(eval_metrics, "eval")
                    eval_next = False

                if self._step > 0:
                    if info.get("terminated", False) and not get_config_value(
                        self.cfg, "algorithm.episodic"
                    ):
                        raise ValueError(
                            "Termination detected but episodic mode is off. "
                            "Set ``episodic=True`` to enable terminations."
                        )
                    train_metrics.update(
                        episode_reward=torch.stack(
                            [td["reward"] for td in self._tds[1:]]
                        ).sum(),
                        episode_success=float(info.get("success", 0.0)),
                        episode_length=len(self._tds),
                        episode_terminated=info.get("terminated", False),
                    )
                    train_metrics.update(self._common_metrics())
                    self._log(train_metrics, "train")
                    self._ep_idx = self.buffer.add(torch.cat(self._tds))

                obs = self._reset_env()
                self._tds = [self._to_td(obs)]

            # Collect experience
            collect_start = time()
            if self._step > self.seed_steps:
                action = self.agent.act(obs, t0=len(self._tds) == 1)
            else:
                action = self._random_action()
            obs, reward, done, info = self._step_env(action)
            self._tds.append(self._to_td(obs, action, reward, info.get("terminated")))
            collect_time = time() - collect_start

            # Update agent
            learn_time = 0.0
            if self._step >= self.seed_steps:
                if self._step == self.seed_steps:
                    num_updates = self.seed_steps
                    print("Pretraining agent on seed data...")
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

            self._step += 1
            if self.cfg.log_freq > 0 and self._step >= next_log_step:
                log_metrics = dict(train_metrics)
                if len(self._tds) > 1:
                    log_metrics.setdefault(
                        "episode_reward",
                        torch.stack([td["reward"] for td in self._tds[1:]]).sum(),
                    )
                    log_metrics.setdefault("episode_length", len(self._tds))
                    log_metrics.setdefault(
                        "episode_success",
                        float(info.get("success", 0.0)),
                    )
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

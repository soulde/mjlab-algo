"""TD-MPC2 training runner.

Encapsulates the online training loop: environment interaction,
experience collection, agent updates, evaluation, and logging.
"""

from __future__ import annotations

from pathlib import Path
from time import time
from typing import TYPE_CHECKING

import numpy as np
import torch

from mmrl.logging import format_training_log
from mmrl.runners.model_based import ModelBasedRunner
from mmrl.tdmpc2.buffer import Buffer
from mmrl.env_wrappers.mjlab import MJLabSingleEnvWrapper

if TYPE_CHECKING:
    from mmrl.tdmpc2.tdmpc2 import TDMPC2


class TDMPC2Runner(ModelBasedRunner):
    """Runner for online single-task TD-MPC2 training.

    Manages the training loop: collects experience, updates
    the agent, evaluates periodically, and logs metrics.
    """

    def __init__(
        self,
        cfg,
        env: MJLabSingleEnvWrapper,
        agent: TDMPC2,
        buffer: Buffer,
        log_dir: Path,
    ):
        self.cfg = cfg
        self.env = env
        self.agent = agent
        self.buffer = buffer
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self._step = 0
        self._ep_idx = 0
        self._start_time = time()
        self._wandb = None

        self._setup_wandb()

    def _setup_wandb(self) -> None:
        """Initialize W&B logging."""
        if not self.cfg.enable_wandb:
            return
        try:
            import wandb

            wandb.init(
                project=self.cfg.wandb_project,
                entity=self.cfg.wandb_entity,
                name=str(self.cfg.seed),
                group=self.cfg.exp_name,
                dir=str(self.log_dir),
                config={
                    k: v for k, v in self.cfg.__dict__.items() if not k.startswith("_")
                },
                settings=wandb.Settings(silent=True) if self.cfg.wandb_silent else None,
            )
            self._wandb = wandb
        except Exception:
            self._wandb = None

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
        """Log metrics to console and optionally W&B."""
        if self._wandb is not None:
            _d = {f"{category}/{k}": v for k, v in metrics.items()}
            step = metrics.get("step", self._step)
            self._wandb.log(_d, step=step)

        step = int(metrics.get("step", self._step))
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
            action = torch.full_like(self.env.rand_act(), float("nan"))
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

    def eval(self) -> dict:
        """Evaluate agent for ``eval_episodes`` episodes."""
        ep_rewards: list[float] = []
        ep_successes: list[float] = []
        ep_lengths: list[int] = []
        if self.cfg.eval_episodes <= 0:
            return {}

        for _i in range(self.cfg.eval_episodes):
            obs = self.env.reset()
            done = False
            info: dict = {}
            ep_reward = 0.0
            t = 0
            while not done:
                action = self.agent.act(obs, t0=t == 0, eval_mode=True)
                obs, reward, done, info = self.env.step(action)
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

    def train(self) -> None:
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
                    if info.get("terminated", False) and not self.cfg.episodic:
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

                obs = self.env.reset()
                self._tds = [self._to_td(obs)]

            # Collect experience
            collect_start = time()
            if self._step > self.cfg.seed_steps:
                action = self.agent.act(obs, t0=len(self._tds) == 1)
            else:
                action = self.env.rand_act()
            obs, reward, done, info = self.env.step(action)
            self._tds.append(self._to_td(obs, action, reward, info.get("terminated")))
            collect_time = time() - collect_start

            # Update agent
            learn_time = 0.0
            if self._step >= self.cfg.seed_steps:
                if self._step == self.cfg.seed_steps:
                    num_updates = self.cfg.seed_steps
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
            self.agent.save(str(save_path))
            print(f"Saved final model to {save_path}")

        if self._wandb is not None:
            self._wandb.finish()

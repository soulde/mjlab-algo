"""TD-MPC2 training runner.

Encapsulates the online training loop: environment interaction,
experience collection, agent updates, evaluation, and logging.
"""

from pathlib import Path
from time import time

import numpy as np
import torch
from tensordict import TensorDict

from mjlab_algo.tdmpc2.buffer import Buffer
from mjlab_algo.tdmpc2.tdmpc2 import TDMPC2
from mjlab_algo.tdmpc2.vecenv_wrapper import TDMPC2VecEnvWrapper


class TDMPC2Runner:
    """Runner for online single-task TD-MPC2 training.

    Manages the training loop: collects experience, updates
    the agent, evaluates periodically, and logs metrics.
    """

    def __init__(
        self,
        cfg,
        env: TDMPC2VecEnvWrapper,
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
                settings=wandb.Settings(silent="true")
                if self.cfg.wandb_silent
                else None,
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

        # Print key metrics
        if category == "eval":
            rew = metrics.get("episode_reward", 0.0)
            print(f"  Eval | step: {metrics.get('step', 0):,} | reward: {rew:.2f}")
        elif category == "train":
            rew = metrics.get("episode_reward", 0.0)
            print(
                f"  Train | step: {metrics.get('step', 0):,} | "
                f"ep: {metrics.get('episode', 0)} | "
                f"reward: {rew:.2f} | "
                f"SPS: {metrics.get('steps_per_second', 0):.0f}"
            )

    def _to_td(
        self,
        obs: torch.Tensor,
        action: torch.Tensor | None = None,
        reward: torch.Tensor | None = None,
        terminated: torch.Tensor | None = None,
    ) -> TensorDict:
        """Create a TensorDict for a single timestep."""
        if action is None:
            action = torch.full_like(self.env.rand_act(), float("nan"))
        if reward is None:
            reward = torch.tensor(float("nan"))
        if terminated is None:
            terminated = torch.tensor(float("nan"))
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

        for _i in range(self.cfg.eval_episodes):
            obs = self.env.reset()
            done = False
            ep_reward = 0.0
            t = 0
            while not done:
                torch.compiler.cudagraph_mark_step_begin()
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

        print(f"Training on device: {self.agent.device}")
        print(f"Log directory: {self.log_dir}")

        while self._step <= self.cfg.steps:
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
                        episode_reward=torch.tensor(
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
            if self._step > self.cfg.seed_steps:
                action = self.agent.act(obs, t0=len(self._tds) == 1)
            else:
                action = self.env.rand_act()
            obs, reward, done, info = self.env.step(action)
            self._tds.append(self._to_td(obs, action, reward, info.get("terminated")))

            # Update agent
            if self._step >= self.cfg.seed_steps:
                if self._step == self.cfg.seed_steps:
                    num_updates = self.cfg.seed_steps
                    print("Pretraining agent on seed data...")
                else:
                    num_updates = 1
                for _ in range(num_updates):
                    _train_metrics = self.agent.update(self.buffer)
                train_metrics.update(_train_metrics)

            self._step += 1

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

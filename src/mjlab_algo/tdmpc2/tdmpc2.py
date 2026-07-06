"""TD-MPC2 agent.

Implements training + inference for model-based RL with:
- Learned implicit world model (encoder, dynamics, reward, termination, Q)
- MPPI planning in latent space
- Distributional value learning (two-hot encoded)
- Policy prior with entropy regularization
"""

import torch
import torch.nn.functional as F
from tensordict import TensorDict

from mjlab_algo.tdmpc2 import math
from mjlab_algo.tdmpc2.compile import configure_tdmpc2_compile
from mjlab_algo.tdmpc2.layers import api_model_conversion
from mjlab_algo.tdmpc2.scale import RunningScale
from mjlab_algo.tdmpc2.world_model import WorldModel


class TDMPC2(torch.nn.Module):
    """TD-MPC2 agent for single-task and multi-task RL.

    Supports state and pixel observations, with optional MPPI planning.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = WorldModel(cfg).to(self.device)

        param_groups: list[dict] = [
            {
                "params": self.model._encoder.parameters(),
                "lr": self.cfg.lr * self.cfg.enc_lr_scale,
            },
            {"params": self.model._dynamics.parameters()},
            {"params": self.model._reward.parameters()},
        ]
        if self.cfg.episodic:
            param_groups.append({"params": self.model._termination.parameters()})
        param_groups.append({"params": self.model._Qs.parameters()})
        if self.cfg.multitask:
            param_groups.append({"params": self.model._task_emb.parameters()})

        self.optim = torch.optim.Adam(param_groups, lr=self.cfg.lr, capturable=True)
        self.pi_optim = torch.optim.Adam(
            self.model._pi.parameters(),
            lr=self.cfg.lr,
            eps=1e-5,
            capturable=True,
        )
        self.model.eval()
        self.scale = RunningScale(cfg)
        self.cfg.iterations += 2 * int(
            cfg.action_dim >= 20
        )  # Heuristic for large action spaces
        self.discount = (
            torch.tensor(
                [self._get_discount(ep_len) for ep_len in cfg.episode_lengths],
                device=self.device,
            )
            if self.cfg.multitask
            else self._get_discount(cfg.episode_length)
        )
        print("Episode length:", cfg.episode_length)
        print("Discount factor:", self.discount)
        self._prev_mean = torch.nn.Buffer(
            torch.zeros(self.cfg.horizon, self.cfg.action_dim, device=self.device)
        )
        if cfg.compile and torch.cuda.is_available():
            configure_tdmpc2_compile(enabled=True)
            print("Compiling update function with torch.compile...")
            self._update = torch.compile(self._update, mode="reduce-overhead")

    @property
    def plan(self):
        _plan_val = getattr(self, "_plan_val", None)
        if _plan_val is not None:
            return _plan_val
        if self.cfg.compile and torch.cuda.is_available():
            configure_tdmpc2_compile(enabled=True)
            plan = torch.compile(self._plan, mode="reduce-overhead")
        else:
            plan = self._plan
        self._plan_val = plan
        return self._plan_val

    def _get_discount(self, episode_length: int) -> float:
        """Return discount factor for a given episode length.

        Scales linearly with episode length using discount_denom.
        """
        frac = episode_length / self.cfg.discount_denom
        return min(
            max((frac - 1) / frac, self.cfg.discount_min),
            self.cfg.discount_max,
        )

    def save(self, fp: str) -> None:
        """Save state dict of the agent to filepath."""
        torch.save({"model": self.model.state_dict()}, fp)

    def load(self, fp) -> None:
        """Load a saved state dict from filepath (or dictionary)."""
        if isinstance(fp, dict):
            state_dict = fp
        else:
            state_dict = torch.load(
                fp,
                map_location=torch.get_default_device(),
                weights_only=False,
            )
        state_dict = state_dict["model"] if "model" in state_dict else state_dict
        state_dict = api_model_conversion(self.model.state_dict(), state_dict)
        self.model.load_state_dict(state_dict)

    @torch.no_grad()
    def act(
        self,
        obs: torch.Tensor,
        t0: bool = False,
        eval_mode: bool = False,
        task=None,
    ) -> torch.Tensor:
        """Select action by planning in latent space of the world model.

        Args:
            obs: Observation from the environment.
            t0: Whether this is the first observation in the episode.
            eval_mode: If True, use mean of action distribution.
            task: Task index (multi-task only).

        Returns:
            Action to take in the environment.
        """
        obs = obs.to(self.device, non_blocking=True).unsqueeze(0)
        if task is not None:
            task = torch.tensor([task], device=self.device)
        if self.cfg.mpc:
            return self.plan(obs, t0=t0, eval_mode=eval_mode, task=task).cpu()
        z = self.model.encode(obs, task)
        action, info = self.model.pi(z, task)
        if eval_mode:
            action = info["mean"]
        return action[0].cpu()

    @torch.no_grad()
    def _estimate_value(
        self, z: torch.Tensor, actions: torch.Tensor, task
    ) -> torch.Tensor:
        """Estimate value of a trajectory from latent state ``z``."""
        G, discount = 0, 1
        termination = torch.zeros(
            self.cfg.num_samples, 1, dtype=torch.float32, device=z.device
        )
        for t in range(self.cfg.horizon):
            reward = math.two_hot_inv(self.model.reward(z, actions[t], task), self.cfg)
            z = self.model.next(z, actions[t], task)
            G = G + discount * (1 - termination) * reward
            discount_update = (
                self.discount[torch.tensor(task)]
                if self.cfg.multitask
                else self.discount
            )
            discount = discount * discount_update
            if self.cfg.episodic:
                termination = torch.clip(
                    termination + (self.model.termination(z, task) > 0.5).float(),
                    max=1.0,
                )
        action, _ = self.model.pi(z, task)
        return G + discount * (1 - termination) * self.model.Q(
            z, action, task, return_type="avg"
        )

    @torch.no_grad()
    def _plan(
        self,
        obs: torch.Tensor,
        t0: bool = False,
        eval_mode: bool = False,
        task=None,
    ) -> torch.Tensor:
        """Plan action sequence using MPPI in latent space.

        Args:
            obs: Observation to plan from.
            t0: Whether this is the first observation in the episode.
            eval_mode: If True, use mean of action distribution.
            task: Task index (multi-task only).

        Returns:
            Selected action.
        """
        z = self.model.encode(obs, task)
        if self.cfg.num_pi_trajs > 0:
            pi_actions = torch.empty(
                self.cfg.horizon,
                self.cfg.num_pi_trajs,
                self.cfg.action_dim,
                device=self.device,
            )
            _z = z.repeat(self.cfg.num_pi_trajs, 1)
            for t in range(self.cfg.horizon - 1):
                pi_actions[t], _ = self.model.pi(_z, task)
                _z = self.model.next(_z, pi_actions[t], task)
            pi_actions[-1], _ = self.model.pi(_z, task)

        z = z.repeat(self.cfg.num_samples, 1)
        mean = torch.zeros(
            self.cfg.horizon,
            self.cfg.action_dim,
            device=self.device,
        )
        std = torch.full(
            (self.cfg.horizon, self.cfg.action_dim),
            self.cfg.max_std,
            dtype=torch.float,
            device=self.device,
        )
        if not t0:
            mean[:-1] = self._prev_mean[1:]
        actions = torch.empty(
            self.cfg.horizon,
            self.cfg.num_samples,
            self.cfg.action_dim,
            device=self.device,
        )
        if self.cfg.num_pi_trajs > 0:
            actions[:, : self.cfg.num_pi_trajs] = pi_actions

        for _ in range(self.cfg.iterations):
            r = torch.randn(
                self.cfg.horizon,
                self.cfg.num_samples - self.cfg.num_pi_trajs,
                self.cfg.action_dim,
                device=std.device,
            )
            actions_sample = mean.unsqueeze(1) + std.unsqueeze(1) * r
            actions_sample = actions_sample.clamp(-1, 1)
            actions[:, self.cfg.num_pi_trajs :] = actions_sample
            if self.cfg.multitask:
                actions = actions * self.model._action_masks[task]

            value = self._estimate_value(z, actions, task).nan_to_num(0)
            elite_idxs = torch.topk(
                value.squeeze(1), self.cfg.num_elites, dim=0
            ).indices
            elite_value = value[elite_idxs]
            elite_actions = actions[:, elite_idxs]

            max_value = elite_value.max(0).values
            score = torch.exp(self.cfg.temperature * (elite_value - max_value))
            score = score / score.sum(0)
            mean = (score.unsqueeze(0) * elite_actions).sum(dim=1) / (
                score.sum(0) + 1e-9
            )
            std = (
                (score.unsqueeze(0) * (elite_actions - mean.unsqueeze(1)) ** 2).sum(
                    dim=1
                )
                / (score.sum(0) + 1e-9)
            ).sqrt()
            std = std.clamp(self.cfg.min_std, self.cfg.max_std)
            if self.cfg.multitask:
                mean = mean * self.model._action_masks[task]
                std = std * self.model._action_masks[task]

        rand_idx = math.gumbel_softmax_sample(score.squeeze(1))
        actions = torch.index_select(elite_actions, 1, rand_idx).squeeze(1)
        a, std = actions[0], std[0]
        if not eval_mode:
            a = a + std * torch.randn(self.cfg.action_dim, device=std.device)
        self._prev_mean.copy_(mean)
        return a.clamp(-1, 1)

    def update_pi(self, zs: torch.Tensor, task) -> TensorDict:
        """Update policy using a sequence of latent states.

        Args:
            zs: Sequence of latent states (horizon+1, batch_size, latent_dim).
            task: Task index (multi-task only).

        Returns:
            Dict of training statistics.
        """
        action, info = self.model.pi(zs, task)
        qs = self.model.Q(zs, action, task, return_type="avg", detach=True)
        self.scale.update(qs[0])
        qs = self.scale(qs)

        rho = torch.pow(self.cfg.rho, torch.arange(len(qs), device=self.device))
        pi_loss = (
            (-(self.cfg.entropy_coef * info["scaled_entropy"] + qs)).mean(dim=(1, 2))
            * rho
        ).mean()
        pi_loss.backward()
        pi_grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model._pi.parameters(), self.cfg.grad_clip_norm
        )
        self.pi_optim.step()
        self.pi_optim.zero_grad(set_to_none=True)

        info = TensorDict(
            {
                "pi_loss": pi_loss,
                "pi_grad_norm": pi_grad_norm,
                "pi_entropy": info["entropy"],
                "pi_scaled_entropy": info["scaled_entropy"],
                "pi_scale": self.scale.value,
            }
        )
        return info

    @torch.no_grad()
    def _td_target(
        self,
        next_z: torch.Tensor,
        reward: torch.Tensor,
        terminated: torch.Tensor,
        task,
    ) -> torch.Tensor:
        """Compute TD-target from reward and next latent state."""
        action, _ = self.model.pi(next_z, task)
        discount = (
            self.discount[task].unsqueeze(-1) if self.cfg.multitask else self.discount
        )
        return reward + discount * (1 - terminated) * self.model.Q(
            next_z, action, task, return_type="min", target=True
        )

    def _update(self, obs, action, reward, terminated, task=None):
        # Compute targets
        with torch.no_grad():
            next_z = self.model.encode(obs[1:], task)
            td_targets = self._td_target(next_z, reward, terminated, task)

        self.model.train()

        # Latent rollout
        zs = torch.empty(
            self.cfg.horizon + 1,
            self.cfg.batch_size,
            self.cfg.latent_dim,
            device=self.device,
        )
        z = self.model.encode(obs[0], task)
        zs[0] = z
        consistency_loss = 0
        for t, (_action, _next_z) in enumerate(
            zip(action.unbind(0), next_z.unbind(0), strict=False)
        ):
            z = self.model.next(z, _action, task)
            consistency_loss = (
                consistency_loss + F.mse_loss(z, _next_z) * self.cfg.rho**t
            )
            zs[t + 1] = z

        # Predictions
        _zs = zs[:-1]
        qs = self.model.Q(_zs, action, task, return_type="all")
        reward_preds = self.model.reward(_zs, action, task)
        if self.cfg.episodic:
            termination_pred = self.model.termination(zs[1:], task, unnormalized=True)

        # Compute losses
        reward_loss = 0.0
        value_loss = 0.0
        for t, (
            rew_pred_unbind,
            rew_unbind,
            td_targets_unbind,
            qs_unbind,
        ) in enumerate(
            zip(
                reward_preds.unbind(0),
                reward.unbind(0),
                td_targets.unbind(0),
                qs.unbind(1),
                strict=False,
            )
        ):
            reward_loss = (
                reward_loss
                + math.soft_ce(rew_pred_unbind, rew_unbind, self.cfg).mean()
                * self.cfg.rho**t
            )
            for _, qs_unbind_unbind in enumerate(qs_unbind.unbind(0)):
                value_loss = (
                    value_loss
                    + math.soft_ce(qs_unbind_unbind, td_targets_unbind, self.cfg).mean()
                    * self.cfg.rho**t
                )

        consistency_loss = consistency_loss / self.cfg.horizon
        reward_loss = reward_loss / self.cfg.horizon
        if self.cfg.episodic:
            termination_loss = F.binary_cross_entropy_with_logits(
                termination_pred, terminated
            )
        else:
            termination_loss = torch.tensor(0.0, device=self.device)
        value_loss = value_loss / (self.cfg.horizon * self.cfg.num_q)
        total_loss = (
            self.cfg.consistency_coef * consistency_loss
            + self.cfg.reward_coef * reward_loss
            + self.cfg.termination_coef * termination_loss
            + self.cfg.value_coef * value_loss
        )

        # Update model
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.cfg.grad_clip_norm
        )
        self.optim.step()
        self.optim.zero_grad(set_to_none=True)

        # Update policy
        pi_info = self.update_pi(zs.detach(), task)

        # Update target Q-functions
        self.model.soft_update_target_Q()

        # Return training statistics
        self.model.eval()
        info = TensorDict(
            {
                "consistency_loss": consistency_loss,
                "reward_loss": reward_loss,
                "value_loss": value_loss,
                "termination_loss": termination_loss,
                "total_loss": total_loss,
                "grad_norm": grad_norm,
            }
        )
        if self.cfg.episodic:
            info.update(
                math.termination_statistics(
                    torch.sigmoid(termination_pred[-1]), terminated[-1]
                )
            )
        info.update(pi_info)
        return info.detach().mean()

    def update(self, buffer) -> dict:
        """Main update. Corresponds to one iteration of model learning.

        Args:
            buffer: Replay buffer.

        Returns:
            Dictionary of training statistics.
        """
        obs, action, reward, terminated, task = buffer.sample()
        kwargs = {}
        if task is not None:
            kwargs["task"] = task
        torch.compiler.cudagraph_mark_step_begin()
        return self._update(obs, action, reward, terminated, **kwargs)

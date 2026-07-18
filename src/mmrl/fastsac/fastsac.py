"""FastSAC agent implementation."""

from pathlib import Path

import torch
import torch.nn.functional as F

from mmrl.fastsac.config import FastSACConfig
from mmrl.fastsac.networks import (
    SquashedGaussianActor,
    TwinQNetwork,
    init_weights,
)
from mmrl.memories import OffPolicyBatch


class FastSAC:
    """Soft Actor-Critic agent for flat continuous-control observations."""

    def __init__(self, cfg: FastSACConfig):
        if cfg.obs_dim <= 0 or cfg.action_dim <= 0:
            raise ValueError("FastSACConfig.obs_dim and action_dim must be set.")
        self.cfg = cfg
        self.device = torch.device(
            cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        )
        self.actor = SquashedGaussianActor(
            cfg.obs_dim,
            cfg.action_dim,
            cfg.hidden_dims,
            cfg.log_std_min,
            cfg.log_std_max,
        ).to(self.device)
        self.critic = TwinQNetwork(cfg.obs_dim, cfg.action_dim, cfg.hidden_dims).to(
            self.device
        )
        self.target_critic = TwinQNetwork(
            cfg.obs_dim, cfg.action_dim, cfg.hidden_dims
        ).to(self.device)
        self.actor.apply(init_weights)
        self.critic.apply(init_weights)
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)
        self.log_alpha = torch.tensor(
            float(torch.log(torch.tensor(cfg.init_alpha))),
            device=self.device,
            requires_grad=True,
        )
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=cfg.alpha_lr)
        self.target_entropy = cfg.target_entropy
        if self.target_entropy is None:
            self.target_entropy = -float(cfg.action_dim)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    @torch.no_grad()
    def act(self, obs: torch.Tensor, eval_mode: bool = False) -> torch.Tensor:
        obs = obs.to(self.device, dtype=torch.float32)
        if eval_mode:
            action = self.actor.deterministic(obs)
        else:
            action, _ = self.actor.sample(obs)
        return action.cpu()

    def update(self, batch: OffPolicyBatch) -> dict[str, float]:
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(batch.next_obs)
            target_q1, target_q2 = self.target_critic(batch.next_obs, next_action)
            target_q = (
                torch.min(target_q1, target_q2) - self.alpha.detach() * next_log_prob
            )
            target = batch.reward + self.cfg.gamma * (1.0 - batch.done) * target_q

        q1, q2 = self.critic(batch.obs, batch.action)
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.critic_optim.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optim.step()

        new_action, log_prob = self.actor.sample(batch.obs)
        new_q1, new_q2 = self.critic(batch.obs, new_action)
        actor_loss = (self.alpha.detach() * log_prob - torch.min(new_q1, new_q2)).mean()
        self.actor_optim.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optim.step()

        if self.cfg.auto_entropy:
            alpha_loss = -(
                self.log_alpha * (log_prob + self.target_entropy).detach()
            ).mean()
            self.alpha_optim.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_optim.step()
        else:
            alpha_loss = torch.zeros((), device=self.device)

        self._soft_update_target()
        return {
            "actor_loss": float(actor_loss.detach().cpu()),
            "critic_loss": float(critic_loss.detach().cpu()),
            "alpha_loss": float(alpha_loss.detach().cpu()),
            "alpha": float(self.alpha.detach().cpu()),
        }

    def _soft_update_target(self) -> None:
        with torch.no_grad():
            for param, target_param in zip(
                self.critic.parameters(), self.target_critic.parameters(), strict=True
            ):
                target_param.data.mul_(1.0 - self.cfg.tau)
                target_param.data.add_(self.cfg.tau * param.data)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "target_critic": self.target_critic.state_dict(),
                "log_alpha": self.log_alpha.detach().cpu(),
                "cfg": self.cfg,
            },
            path,
        )

    def load(self, path: str | Path) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.target_critic.load_state_dict(checkpoint["target_critic"])
        self.log_alpha.data.copy_(checkpoint["log_alpha"].to(self.device))

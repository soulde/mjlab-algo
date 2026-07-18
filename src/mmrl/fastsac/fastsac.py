"""FastSAC agent implementation."""

from pathlib import Path

import torch
import torch.nn.functional as F

from mmrl.config import get_config_value
from mmrl.models import (
    SquashedGaussianActor,
    TwinQNetwork,
    init_weights,
)
from mmrl.memories import OffPolicyBatch


class FastSAC:
    """Soft Actor-Critic agent for flat continuous-control observations."""

    def __init__(self, cfg, obs_dim: int, action_dim: int, device: torch.device):
        self.cfg = cfg
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.device = torch.device(device)
        self.actor = SquashedGaussianActor(
            self.obs_dim,
            self.action_dim,
            tuple(get_config_value(cfg, "actor.hidden_dims")),
            get_config_value(cfg, "actor.log_std_min"),
            get_config_value(cfg, "actor.log_std_max"),
        ).to(self.device)
        critic_dims = tuple(get_config_value(cfg, "critic.hidden_dims"))
        self.critic = TwinQNetwork(self.obs_dim, self.action_dim, critic_dims).to(self.device)
        self.target_critic = TwinQNetwork(
            self.obs_dim, self.action_dim, critic_dims
        ).to(self.device)
        self.actor.apply(init_weights)
        self.critic.apply(init_weights)
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_optim = torch.optim.Adam(
            self.actor.parameters(), lr=get_config_value(cfg, "algorithm.actor_lr")
        )
        self.critic_optim = torch.optim.Adam(
            self.critic.parameters(), lr=get_config_value(cfg, "algorithm.critic_lr")
        )
        self.log_alpha = torch.tensor(
            float(torch.log(torch.tensor(get_config_value(cfg, "algorithm.init_alpha")))),
            device=self.device,
            requires_grad=True,
        )
        self.alpha_optim = torch.optim.Adam(
            [self.log_alpha], lr=get_config_value(cfg, "algorithm.alpha_lr")
        )
        self.target_entropy = get_config_value(cfg, "algorithm.target_entropy")
        if self.target_entropy is None:
            self.target_entropy = -float(self.action_dim)

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
            target = batch.reward + get_config_value(
                self.cfg, "algorithm.gamma"
            ) * (1.0 - batch.done) * target_q

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

        if get_config_value(self.cfg, "algorithm.auto_entropy"):
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
                tau = get_config_value(self.cfg, "algorithm.tau")
                target_param.data.mul_(1.0 - tau)
                target_param.data.add_(tau * param.data)

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

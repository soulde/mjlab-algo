"""Continuous-action actor-critic model for PPO.

The design follows RSL-RL's actor-critic module under the BSD-3-Clause
license, adapted to mmrl's explicit config and model interfaces.
"""

import torch

from mmrl.models import GaussianActor, Model, ValueNetwork


class ActorCritic(Model):
    """Diagonal-Gaussian policy and value function with separate MLPs."""

    def __init__(
        self,
        cfg,
        obs_dim: int,
        action_dim: int,
        critic_obs_dim: int | None = None,
    ):
        super().__init__()
        self.actor = GaussianActor(
            obs_dim,
            action_dim,
            tuple(cfg.actor_hidden_dims),
            cfg.activation,
            cfg.init_noise_std,
            cfg.noise_std_type,
        )
        self.critic_obs_dim = critic_obs_dim or obs_dim
        self.critic = ValueNetwork(
            self.critic_obs_dim,
            tuple(cfg.critic_hidden_dims),
            cfg.activation,
        )

    @property
    def log_std(self) -> torch.nn.Parameter:
        return self.actor.log_std

    def distribution(self, obs: torch.Tensor) -> torch.distributions.Normal:
        return self.actor.distribution(obs)

    def act(
        self,
        obs: torch.Tensor,
        critic_obs: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution = self.distribution(obs)
        action = distribution.sample()
        log_prob = distribution.log_prob(action).sum(dim=-1, keepdim=True)
        value_obs = critic_obs if critic_obs is not None else obs
        return action, log_prob, self.critic(value_obs)

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        critic_obs: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution = self.distribution(obs)
        log_prob = distribution.log_prob(action).sum(dim=-1, keepdim=True)
        entropy = distribution.entropy().sum(dim=-1, keepdim=True)
        return (
            log_prob,
            entropy,
            self.critic(critic_obs if critic_obs is not None else obs),
        )

    def act_inference(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor(obs)

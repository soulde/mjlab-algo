"""Continuous-action actor-critic model for PPO.

The design follows RSL-RL's actor-critic module under the BSD-3-Clause
license, adapted to mmrl's explicit config and model interfaces.
"""

import torch
import torch.nn as nn

from mmrl.models import Model


def _activation(name: str) -> type[nn.Module]:
    activations = {
        "elu": nn.ELU,
        "relu": nn.ReLU,
        "selu": nn.SELU,
        "tanh": nn.Tanh,
    }
    try:
        return activations[name.lower()]
    except KeyError as error:
        raise ValueError(f"Unsupported activation {name!r}.") from error


def _mlp(
    input_dim: int,
    hidden_dims: tuple[int, ...],
    output_dim: int,
    activation: str,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    current_dim = input_dim
    activation_type = _activation(activation)
    for hidden_dim in hidden_dims:
        layers.extend((nn.Linear(current_dim, hidden_dim), activation_type()))
        current_dim = hidden_dim
    layers.append(nn.Linear(current_dim, output_dim))
    return nn.Sequential(*layers)


class ActorCritic(Model):
    """Diagonal-Gaussian policy and value function with separate MLPs."""

    def __init__(
        self,
        cfg,
        obs_dim: int,
        action_dim: int,
    ):
        super().__init__()
        self.actor = _mlp(
            obs_dim,
            tuple(cfg.actor_hidden_dims),
            action_dim,
            cfg.activation,
        )
        self.critic = _mlp(
            obs_dim,
            tuple(cfg.critic_hidden_dims),
            1,
            cfg.activation,
        )
        if cfg.noise_std_type == "scalar":
            self.log_std = nn.Parameter(
                torch.tensor(float(cfg.init_noise_std)).log()
            )
        elif cfg.noise_std_type == "per_action":
            self.log_std = nn.Parameter(
                torch.full((action_dim,), float(cfg.init_noise_std)).log()
            )
        else:
            raise ValueError(
                f"Unsupported noise_std_type {cfg.noise_std_type!r}."
            )

    def distribution(self, obs: torch.Tensor) -> torch.distributions.Normal:
        mean = self.actor(obs)
        std = self.log_std.exp().expand_as(mean)
        return torch.distributions.Normal(mean, std)

    def act(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution = self.distribution(obs)
        action = distribution.sample()
        log_prob = distribution.log_prob(action).sum(dim=-1, keepdim=True)
        return action, log_prob, self.critic(obs)

    def evaluate_actions(
        self, obs: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution = self.distribution(obs)
        log_prob = distribution.log_prob(action).sum(dim=-1, keepdim=True)
        entropy = distribution.entropy().sum(dim=-1, keepdim=True)
        return log_prob, entropy, self.critic(obs)

    def act_inference(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor(obs)

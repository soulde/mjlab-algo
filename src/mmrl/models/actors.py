"""Reusable continuous-action actor networks."""

import math

import torch
import torch.nn as nn

from mmrl.models.base import Model
from mmrl.models.mlp import build_mlp


def init_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        bound = 1.0 / math.sqrt(module.weight.shape[0])
        nn.init.uniform_(module.weight, -bound, bound)
        nn.init.uniform_(module.bias, -bound, bound)


class GaussianActor(Model):
    """Unsquashed diagonal-Gaussian policy used by PPO."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: tuple[int, ...],
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
    ):
        super().__init__()
        self.net = build_mlp(obs_dim, hidden_dims, action_dim, activation)
        shape = () if noise_std_type == "scalar" else (action_dim,)
        if noise_std_type not in {"scalar", "per_action"}:
            raise ValueError(f"Unsupported noise_std_type {noise_std_type!r}.")
        self.log_std = nn.Parameter(
            torch.full(shape, float(init_noise_std)).log()
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

    def distribution(self, obs: torch.Tensor) -> torch.distributions.Normal:
        mean = self(obs)
        return torch.distributions.Normal(mean, self.log_std.exp().expand_as(mean))


class SquashedGaussianActor(Model):
    """Tanh-squashed Gaussian policy used by SAC."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: tuple[int, ...],
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
    ):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.net = build_mlp(obs_dim, hidden_dims, 2 * action_dim, "relu")

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.net(obs).chunk(2, dim=-1)
        return mean, log_std.clamp(self.log_std_min, self.log_std_max)

    def sample(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self(obs)
        distribution = torch.distributions.Normal(mean, log_std.exp())
        pre_tanh = distribution.rsample()
        action = torch.tanh(pre_tanh)
        correction = torch.log(1.0 - action.square() + 1e-6)
        log_prob = (distribution.log_prob(pre_tanh) - correction).sum(
            dim=-1, keepdim=True
        )
        return action, log_prob

    def deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        mean, _ = self(obs)
        return torch.tanh(mean)

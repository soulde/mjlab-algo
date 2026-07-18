"""Neural network modules for FastSAC."""

import math

import torch
import torch.nn as nn


def _mlp(in_dim: int, hidden_dims: tuple[int, ...], out_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    last_dim = in_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(last_dim, hidden_dim))
        layers.append(nn.ReLU())
        last_dim = hidden_dim
    layers.append(nn.Linear(last_dim, out_dim))
    return nn.Sequential(*layers)


class SquashedGaussianActor(nn.Module):
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
        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.net = _mlp(obs_dim, hidden_dims, 2 * action_dim)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.net(obs).chunk(2, dim=-1)
        log_std = log_std.clamp(self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self(obs)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        pre_tanh = normal.rsample()
        action = torch.tanh(pre_tanh)
        log_prob = normal.log_prob(pre_tanh)
        correction = torch.log(1.0 - action.pow(2) + 1e-6)
        log_prob = (log_prob - correction).sum(dim=-1, keepdim=True)
        return action, log_prob

    def deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        mean, _ = self(obs)
        return torch.tanh(mean)


class QNetwork(nn.Module):
    """State-action value network."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: tuple[int, ...]):
        super().__init__()
        self.net = _mlp(obs_dim + action_dim, hidden_dims, 1)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, action], dim=-1))


class TwinQNetwork(nn.Module):
    """Twin critic network for clipped double Q-learning."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: tuple[int, ...]):
        super().__init__()
        self.q1 = QNetwork(obs_dim, action_dim, hidden_dims)
        self.q2 = QNetwork(obs_dim, action_dim, hidden_dims)

    def forward(
        self, obs: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.q1(obs, action), self.q2(obs, action)


def init_weights(module: nn.Module) -> None:
    """Initialize linear layers with SAC-friendly defaults."""
    if isinstance(module, nn.Linear):
        bound = 1.0 / math.sqrt(module.weight.shape[0])
        nn.init.uniform_(module.weight, -bound, bound)
        nn.init.uniform_(module.bias, -bound, bound)

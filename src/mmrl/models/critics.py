"""Reusable value and Q-function networks."""

import torch

from mmrl.models.base import Model
from mmrl.models.mlp import build_mlp


class ValueNetwork(Model):
    def __init__(
        self,
        obs_dim: int,
        hidden_dims: tuple[int, ...],
        activation: str = "elu",
    ):
        super().__init__()
        self.net = build_mlp(obs_dim, hidden_dims, 1, activation)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class QNetwork(Model):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: tuple[int, ...]):
        super().__init__()
        self.net = build_mlp(obs_dim + action_dim, hidden_dims, 1, "relu")

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat((obs, action), dim=-1))


class TwinQNetwork(Model):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: tuple[int, ...]):
        super().__init__()
        self.q1 = QNetwork(obs_dim, action_dim, hidden_dims)
        self.q2 = QNetwork(obs_dim, action_dim, hidden_dims)

    def forward(
        self, obs: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.q1(obs, action), self.q2(obs, action)

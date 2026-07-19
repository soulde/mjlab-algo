"""Adversarial Motion Priors model components."""

import torch
import torch.nn as nn
from torch import autograd

from mmrl.models.base import Model


class AMPDiscriminator(Model):
    """LSGAN discriminator over consecutive AMP observations."""

    def __init__(
        self,
        observation_dim: int,
        hidden_dims: tuple[int, ...],
    ) -> None:
        super().__init__()
        if not hidden_dims:
            raise ValueError("AMP discriminator requires at least one hidden layer.")
        layers: list[nn.Module] = []
        input_dim = observation_dim * 2
        for hidden_dim in hidden_dims:
            layers.extend((nn.Linear(input_dim, hidden_dim), nn.LeakyReLU()))
            input_dim = hidden_dim
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(input_dim, 1)

    def forward(
        self, state: torch.Tensor, next_state: torch.Tensor
    ) -> torch.Tensor:
        return self.head(self.trunk(torch.cat((state, next_state), dim=-1)))

    def gradient_penalty(
        self,
        expert_state: torch.Tensor,
        expert_next_state: torch.Tensor,
    ) -> torch.Tensor:
        expert_input = torch.cat((expert_state, expert_next_state), dim=-1)
        expert_input = expert_input.detach().requires_grad_(True)
        prediction = self.head(self.trunk(expert_input))
        gradient = autograd.grad(
            prediction,
            expert_input,
            grad_outputs=torch.ones_like(prediction),
            create_graph=True,
            only_inputs=True,
        )[0]
        return gradient.square().sum(dim=-1).mean()

    @torch.no_grad()
    def style_reward(
        self,
        state: torch.Tensor,
        next_state: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        prediction = self(state, next_state)
        return scale * (1.0 - 0.25 * (prediction - 1.0).square()).clamp_min(0.0)


class RunningMeanStd(nn.Module):
    """Streaming feature normalizer stored with AMP checkpoints."""

    def __init__(self, feature_dim: int, epsilon: float = 1e-4) -> None:
        super().__init__()
        self.register_buffer("mean", torch.zeros(feature_dim))
        self.register_buffer("var", torch.ones(feature_dim))
        self.register_buffer("count", torch.tensor(epsilon))

    def normalize(self, value: torch.Tensor) -> torch.Tensor:
        return (value - self.mean) / self.var.sqrt().clamp_min(1e-6)

    @torch.no_grad()
    def update(self, value: torch.Tensor) -> None:
        value = value.detach().reshape(-1, self.mean.numel())
        batch_count = value.shape[0]
        if batch_count == 0:
            return
        batch_mean = value.mean(dim=0)
        batch_var = value.var(dim=0, unbiased=False)
        delta = batch_mean - self.mean
        total = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total
        m2 = self.var * self.count + batch_var * batch_count
        m2 += delta.square() * self.count * batch_count / total
        self.mean.copy_(new_mean)
        self.var.copy_(m2 / total)
        self.count.copy_(total)

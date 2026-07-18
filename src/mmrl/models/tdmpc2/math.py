"""Math utilities for the TD-MPC2 model and planner.

Includes two-hot encoding, symlog/symexp transforms, Gaussian log-probability,
and other distributional RL helpers.
"""

import torch
import torch.nn.functional as F
from tensordict import TensorDict


def soft_ce(pred: torch.Tensor, target: torch.Tensor, cfg) -> torch.Tensor:
    """Compute cross-entropy loss between predictions and soft targets."""
    pred = F.log_softmax(pred, dim=-1)
    target = two_hot(target, cfg)
    return -(target * pred).sum(-1, keepdim=True)


def log_std(x: torch.Tensor, low: torch.Tensor, dif: torch.Tensor) -> torch.Tensor:
    """Map raw network output to log-std range via tanh."""
    return low + 0.5 * dif * (torch.tanh(x) + 1)


def gaussian_logprob(eps: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
    """Compute Gaussian log-probability."""
    residual = -0.5 * eps.pow(2) - log_std
    log_prob = residual - 0.9189385175704956
    return log_prob.sum(-1, keepdim=True)


def squash(
    mu: torch.Tensor, pi: torch.Tensor, log_pi: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply tanh squashing with log-probability correction."""
    mu = torch.tanh(mu)
    pi = torch.tanh(pi)
    squashed_pi = torch.log(F.relu(1 - pi.pow(2)) + 1e-6)
    log_pi = log_pi - squashed_pi.sum(-1, keepdim=True)
    return mu, pi, log_pi


def int_to_one_hot(x: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Convert integer tensor to one-hot tensor (supports batched inputs)."""
    one_hot = torch.zeros(*x.shape, num_classes, device=x.device)
    one_hot.scatter_(-1, x.unsqueeze(-1), 1)
    return one_hot


def symlog(x: torch.Tensor) -> torch.Tensor:
    """Symmetric logarithmic function (from DreamerV3)."""
    return torch.sign(x) * torch.log(1 + torch.abs(x))


def symexp(x: torch.Tensor) -> torch.Tensor:
    """Symmetric exponential function (from DreamerV3)."""
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)


def two_hot(x: torch.Tensor, cfg) -> torch.Tensor:
    """Convert scalars to soft two-hot encoded targets for discrete regression."""
    if cfg.num_bins == 0:
        return x
    if cfg.num_bins == 1:
        return symlog(x)
    x = torch.clamp(symlog(x), cfg.vmin, cfg.vmax).squeeze(1)
    bin_idx = torch.floor((x - cfg.vmin) / cfg.bin_size)
    bin_offset = ((x - cfg.vmin) / cfg.bin_size - bin_idx).unsqueeze(-1)
    soft_two_hot = torch.zeros(x.shape[0], cfg.num_bins, device=x.device, dtype=x.dtype)
    bin_idx = bin_idx.long()
    soft_two_hot = soft_two_hot.scatter(1, bin_idx.unsqueeze(1), 1 - bin_offset)
    soft_two_hot = soft_two_hot.scatter(
        1, (bin_idx.unsqueeze(1) + 1) % cfg.num_bins, bin_offset
    )
    return soft_two_hot


def two_hot_inv(x: torch.Tensor, cfg) -> torch.Tensor:
    """Convert soft two-hot encoded vectors back to scalars."""
    if cfg.num_bins == 0:
        return x
    if cfg.num_bins == 1:
        return symexp(x)
    dreg_bins = torch.linspace(
        cfg.vmin, cfg.vmax, cfg.num_bins, device=x.device, dtype=x.dtype
    )
    x = F.softmax(x, dim=-1)
    x = torch.sum(x * dreg_bins, dim=-1, keepdim=True)
    return symexp(x)


def gumbel_softmax_sample(
    p: torch.Tensor, temperature: float = 1.0, dim: int = 0
) -> torch.Tensor:
    """Sample from the Gumbel-Softmax distribution."""
    logits = p.log()
    gumbels = (
        -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format)
        .exponential_()
        .log()
    )
    gumbels = (logits + gumbels) / temperature
    y_soft = gumbels.softmax(dim)
    return y_soft.argmax(-1)


def termination_statistics(
    pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-9
) -> TensorDict:
    """Compute episode termination statistics (rate, F1)."""
    pred = pred.squeeze(-1)
    target = target.squeeze(-1)
    rate = target.sum() / len(target)
    tp = ((pred > 0.5) & (target == 1)).sum()
    fn = ((pred <= 0.5) & (target == 1)).sum()
    fp = ((pred > 0.5) & (target == 0)).sum()
    recall = tp / (tp + fn + eps)
    precision = tp / (tp + fp + eps)
    f1 = 2 * (precision * recall) / (precision + recall + eps)
    return TensorDict(
        {
            "termination_rate": rate,
            "termination_f1": f1,
        }
    )

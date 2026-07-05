"""Weight initialization for TD-MPC2."""

import torch.nn as nn


def weight_init(m: nn.Module) -> None:
    """Custom weight initialization for TD-MPC2."""
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Embedding):
        nn.init.uniform_(m.weight, -0.02, 0.02)
    elif isinstance(m, nn.ParameterList):
        for i, p in enumerate(m):
            if p.dim() == 3:  # Linear weight
                nn.init.trunc_normal_(p, std=0.02)
                nn.init.constant_(m[i + 1], 0)  # Bias


def zero_(params: list[nn.Parameter]) -> None:
    """Initialize parameters to zero."""
    for p in params:
        p.data.fill_(0)

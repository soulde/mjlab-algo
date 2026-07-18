"""Base model interfaces."""

from abc import ABC

import torch.nn as nn


class Model(nn.Module, ABC):
    """Base class for neural network modules."""


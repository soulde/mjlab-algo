"""Shared multilayer perceptron building blocks."""

import torch.nn as nn


def activation(name: str) -> type[nn.Module]:
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


def build_mlp(
    input_dim: int,
    hidden_dims: tuple[int, ...],
    output_dim: int,
    activation_name: str = "relu",
) -> nn.Sequential:
    layers: list[nn.Module] = []
    current_dim = input_dim
    activation_type = activation(activation_name)
    for hidden_dim in hidden_dims:
        layers.extend((nn.Linear(current_dim, hidden_dim), activation_type()))
        current_dim = hidden_dim
    layers.append(nn.Linear(current_dim, output_dim))
    return nn.Sequential(*layers)

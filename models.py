"""
Neural network model definitions for DrugReflector.

This module contains the nnFC model architecture and supporting utilities.
"""

from __future__ import annotations
from collections.abc import Iterable
from typing import Any, Mapping

import torch
import torch.nn as nn


def fetch_activation(activation_name: str, activation_init_params: Mapping[str, Any] | None = None) -> nn.Module:
    """Get activation function by a name and initialize it."""
    activation_init_params = {} if activation_init_params is None else activation_init_params
    
    activation_map = {
        'Sigmoid': nn.Sigmoid,
        'ReLU': nn.ReLU,
        'Mish': nn.Mish,
        'Tanh': nn.Tanh,
        'SELU': nn.SELU,
        'LeakyReLU': nn.LeakyReLU,
    }
    
    if activation_name not in activation_map:
        raise ValueError(f"Unknown activation: {activation_name}")
    
    return activation_map[activation_name](**activation_init_params)


def init_weights(m):
    """Initialize weights for linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)


def order_activation_block(order, act_layer, bn_layer, drop_layer):
    """Create activation block with specified order."""
    modules = []
    for layer in order.split('-'):
        if layer == 'act':
            modules.append(act_layer)
        elif layer == 'drop':
            modules.append(drop_layer)
        else:
            modules.append(bn_layer)
    return nn.Sequential(*modules)


def make_fc_encoder(
    input_dim: int,
    hidden_dims: Iterable[int],
    dropout_p: float = 0.25,
    activation: str = 'ReLU',
    batch_norm: bool = True,
    order: str = 'act-drop-bn',
) -> tuple[list[nn.Module], int]:
    """Generate fully connected sequence of layers."""
    input_dim_c = input_dim
    modules = []
    if isinstance(hidden_dims, int):
        hidden_dims = [hidden_dims]
    for h_dim in hidden_dims:
        modules.append(
            nn.Sequential(
                nn.Linear(input_dim_c, h_dim),
                order_activation_block(
                    order,
                    fetch_activation(activation),
                    nn.BatchNorm1d(h_dim) if batch_norm else nn.Identity(),
                    nn.Dropout(dropout_p) if dropout_p > 0 else nn.Identity(),
                ),
            )
        )
        input_dim_c = h_dim
    return modules, input_dim_c


class nnFC(nn.Module):
    """
    Multi Layer Fully Connected Neural Network.

    Parameters
    ----------
    input_dim : int
        Dimension of the input tensor (batch_size * input_dim)
    output_dim : int
        Dimension of the output tensor (batch_size * output_dim)
    dropout_p : float, default=0.25
        Dropout probability after each Linear layer. If set to 0 replaces Dropout with Identity layer
    activation : str, default='ReLU'
        Activation layer name. Currently supported: 'Sigmoid', 'ReLU', 'Mish', 'Tanh', 'SELU', 'LeakyReLU'
    batch_norm : bool, default=True
        If True, applies batchnorm. If False, applies Identity layer
    hidden_dims : List or None, default=None
        Dimensions of hidden layers
    order: str, default='act-drop-bn'
        Order of activation + dropout + batch norm layers
        Options: 'act-drop-bn', 'act-bn-drop', 'bn-drop-act', 'bn-act-drop', 'drop-act-bn', 'drop-bn-act'
    final_layer_bias: bool, default=True
        Whether to have a bias term for the final layer
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        dropout_p: float = 0.25,
        activation: str = 'ReLU',
        batch_norm: bool = True,
        hidden_dims: Iterable[int] | None = None,
        order: str = 'act-drop-bn',
        final_layer_bias: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        input_dim_c = input_dim
        if hidden_dims is not None:
            modules, input_dim_c = make_fc_encoder(
                input_dim, hidden_dims, dropout_p, activation, batch_norm, order
            )
            self.encoder = nn.Sequential(*modules)
            self.encoder.apply(init_weights)
        else:
            self.encoder = nn.Identity()
        self.final_layer = nn.Linear(input_dim_c, output_dim, bias=final_layer_bias)
        self.final_layer.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.final_layer(self.encoder(x))
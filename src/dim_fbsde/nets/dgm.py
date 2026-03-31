"""
Deep Galerkin Method (DGM) Neural Network Architecture.

Implements the LSTM-style network from Sirignano & Spiliopoulos (2018)
for solving high-dimensional PDEs.

References:
- Thesis Chapter 8: The Deep Galerkin Method
- Thesis Section 8.4: Neural Network Architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class LSTMLayer(nn.Module):
    """
    Custom LSTM-like gating layer for the DGM architecture.

    See Thesis Section 8.4, equations for Z_l, G_l, R_l, H_l, S_{l+1}.
    """

    def __init__(self, output_dim: int, input_dim: int,
                 trans1: str = "tanh", trans2: str = "tanh"):
        """
        Args:
            output_dim (int): Number of neurons in this layer.
            input_dim (int): Dimension of the concatenated input [t, x].
            trans1 (str): Activation for the Z, G, R gates. Defaults to 'tanh'.
            trans2 (str): Activation for the H transformation. Defaults to 'tanh'.
        """
        super().__init__()
        act = {"tanh": torch.tanh, "relu": F.relu, "sigmoid": torch.sigmoid}
        self.trans1 = act.get(trans1, torch.tanh)
        self.trans2 = act.get(trans2, torch.tanh)

        self.Uz = nn.Parameter(torch.empty(input_dim, output_dim))
        self.Ug = nn.Parameter(torch.empty(input_dim, output_dim))
        self.Ur = nn.Parameter(torch.empty(input_dim, output_dim))
        self.Uh = nn.Parameter(torch.empty(input_dim, output_dim))
        self.Wz = nn.Parameter(torch.empty(output_dim, output_dim))
        self.Wg = nn.Parameter(torch.empty(output_dim, output_dim))
        self.Wr = nn.Parameter(torch.empty(output_dim, output_dim))
        self.Wh = nn.Parameter(torch.empty(output_dim, output_dim))
        self.bz = nn.Parameter(torch.zeros(output_dim))
        self.bg = nn.Parameter(torch.zeros(output_dim))
        self.br = nn.Parameter(torch.zeros(output_dim))
        self.bh = nn.Parameter(torch.zeros(output_dim))

        for w in [self.Uz, self.Ug, self.Ur, self.Uh,
                  self.Wz, self.Wg, self.Wr, self.Wh]:
            nn.init.xavier_uniform_(w)

    def forward(self, S: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        Z = self.trans1(X @ self.Uz + S @ self.Wz + self.bz)
        G = self.trans1(X @ self.Ug + S @ self.Wg + self.bg)
        R = self.trans1(X @ self.Ur + S @ self.Wr + self.br)
        H = self.trans2(X @ self.Uh + (S * R) @ self.Wh + self.bh)
        return (1 - G) * H + Z * S


class DenseLayer(nn.Module):
    """Single dense layer with optional activation."""

    def __init__(self, output_dim: int, input_dim: int,
                 transformation: Optional[str] = None):
        """
        Args:
            output_dim (int): Number of output features.
            input_dim (int): Number of input features.
            transformation (str, optional): Activation function ('tanh', 'relu', or None).
        """
        super().__init__()
        self.W = nn.Parameter(torch.empty(input_dim, output_dim))
        self.b = nn.Parameter(torch.zeros(output_dim))
        act = {"tanh": torch.tanh, "relu": F.relu}
        self.transformation = act.get(transformation, None)
        nn.init.xavier_uniform_(self.W)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        S = X @ self.W + self.b
        return self.transformation(S) if self.transformation else S


class DGMNet(nn.Module):
    """
    Deep Galerkin Method network.

    As described in Thesis Section 8.4, this network uses an LSTM-style gating
    architecture to approximate PDE solutions. The network takes a concatenated
    time-space input vector (t, x) and produces a scalar output approximating u(t, x).

    Architecture: Initial dense layer -> L LSTM layers -> Output dense layer.
    See Thesis Figure 8.1.
    """

    def __init__(self, layer_width: int, n_layers: int, input_dim: int,
                 final_trans: Optional[str] = None):
        """
        Initializes the DGM network.

        Args:
            layer_width (int): Number of neurons per layer (M in the thesis).
            n_layers (int): Number of LSTM layers (L in the thesis).
            input_dim (int): Spatial dimension (dim_x). The network expects a concatenated
                             input of shape [batch, 1 + dim_x], where the first column is time.
            final_trans (str, optional): Activation for the output layer.
                                         None for linear output. Defaults to None.
        """
        super().__init__()
        self.input_dim = input_dim
        self.initial_layer = DenseLayer(layer_width, input_dim + 1,
                                        transformation="tanh")
        self.LSTMLayerList = nn.ModuleList(
            [LSTMLayer(layer_width, input_dim + 1) for _ in range(n_layers)]
        )
        self.final_layer = DenseLayer(1, layer_width, transformation=final_trans)

    def forward(self, tx: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            tx (Tensor): Input tensor of shape [Batch, 1 + dim_x].
                         Represents the concatenated time and state vector (t, X_t).

        Returns:
            Tensor: Output tensor of shape [Batch, 1].
                    Represents the approximation of the PDE solution u(t, x).
        """
        S = self.initial_layer(tx)
        for lstm in self.LSTMLayerList:
            S = lstm(S, tx)
        return self.final_layer(S)

    def __repr__(self):
        return (f"DGMNet(width={self.initial_layer.W.shape[1]}, "
                f"layers={len(self.LSTMLayerList)}, "
                f"input_dim={self.input_dim})")

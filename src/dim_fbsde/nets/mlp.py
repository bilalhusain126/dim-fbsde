"""
Multi-Layer Perceptron (MLP) architecture.

This module defines the feedforward neural network used as the function approximator 
for the conditional expectations in the Deep Iterative Method.

References:
- Thesis Section 2.6: Neural Networks as Function Approximators
- Thesis Section 4.2.1: Model Architecture
"""

import torch
import torch.nn as nn
from typing import List

class MLP(nn.Module):
    """
    A standard Multi-Layer Perceptron (Feed-Forward Neural Network).
    
    As described in Thesis Section 2.6.1, this network is constructed dynamically 
    from a list of hidden layer dimensions. It approximates the solution function 
    u(t, x) by taking a concatenated input vector (time + state) and producing 
    the vector-valued process Y or Z.
    """

    def __init__(self, 
                 input_dim: int, 
                 output_dim: int, 
                 hidden_dims: List[int], 
                 activation: str = 'SiLU'):
        """
        Initializes the MLP parameters.

        Args:
            input_dim (int): Dimension of the input layer. Typically (dim_x + 1) for (t, X_t).
            output_dim (int): Dimension of the output layer. Typically dim_y (for Y) or 
                              dim_y * dim_w (for Z).
            hidden_dims (List[int]): A list containing the number of neurons in each hidden layer.
                                     (e.g., [64, 64, 64]).
            activation (str): The name of the activation function to use. 
                              Options: 'SiLU', 'ReLU', 'Tanh', 'Sigmoid', 'GELU'.
                              Defaults to 'SiLU' (Sigmoid Linear Unit).
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.activation_name = activation
        
        # Resolve activation function factory
        self.activation_fn = self._get_activation(activation)

        layers = []
        
        # Input layer
        if len(hidden_dims) > 0:
            layers.append(nn.Linear(self.input_dim, self.hidden_dims[0]))
            layers.append(self.activation_fn)
            
            # Hidden layers
            for i in range(len(self.hidden_dims) - 1):
                layers.append(nn.Linear(self.hidden_dims[i], self.hidden_dims[i+1]))
                layers.append(self.activation_fn)
            
            # Output layer (Linear projection, no activation)
            layers.append(nn.Linear(self.hidden_dims[-1], self.output_dim))
        else:
            # Linear model case (no hidden layers)
            layers.append(nn.Linear(self.input_dim, self.output_dim))
        
        self.model = nn.Sequential(*layers)

    def _get_activation(self, name: str) -> nn.Module:
        """
        Factory method for activation functions.
        """
        activations = {
            'SiLU': nn.SiLU(),
            'ReLU': nn.ReLU(),
            'Tanh': nn.Tanh(),
            'Sigmoid': nn.Sigmoid(),
            'GELU': nn.GELU()
        }
        if name not in activations:
            raise NotImplementedError(f"Activation '{name}' is not supported. Choose from {list(activations.keys())}")
        return activations[name]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x (Tensor): Input tensor of shape [Batch, input_dim].
                        Represents the concatenated time and state vector (t, X_t).
                        
        Returns:
            Tensor: Output tensor of shape [Batch, output_dim].
                    Represents the approximation of Y_t or Z_t.
        """
        return self.model(x)

    def __repr__(self):
        return (f"MLP(in={self.input_dim}, out={self.output_dim}, "
                f"hidden={self.hidden_dims}, act={self.activation_name})")
        
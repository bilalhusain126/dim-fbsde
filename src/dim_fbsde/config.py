"""
Configuration dataclasses for the DIM and DGM FBSDE solvers.
"""

from dataclasses import dataclass, field
from typing import Literal, Tuple

@dataclass
class TrainingConfig:
    """
    Hyperparameters for the neural network training loop.
    """
    batch_size: int = 500
    epochs: int = 5          # Gradient descent steps per Picard iteration
    learning_rate: float = 1e-4

    lr_decay_step: int = 1000
    lr_decay_rate: float = 0.95

    verbose: bool = True
    gradient_clip_val: float = 1.0

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.epochs <= 0:
            raise ValueError(f"epochs must be positive, got {self.epochs}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        if self.lr_decay_step <= 0:
            raise ValueError(f"lr_decay_step must be positive, got {self.lr_decay_step}")
        if not 0 < self.lr_decay_rate <= 1:
            raise ValueError(f"lr_decay_rate must be in (0, 1], got {self.lr_decay_rate}")
        if self.gradient_clip_val < 0:
            raise ValueError(f"gradient_clip_val must be non-negative, got {self.gradient_clip_val}")


@dataclass
class SolverConfig:
    """
    Hyperparameters for the numerical solver physics and methods.
    """
    T: float = 1.0
    N: int = 120

    num_paths: int = 2000

    # INNER Iterations: How many times the Uncoupled solver refines Y/Z given a fixed X path.
    picard_iterations: int = 10

    # OUTER Iterations: How many times the Coupled solver alternates between simulating X and solving Y/Z.
    global_iterations: int = 10

    z_method: Literal['gradient', 'regression'] = 'gradient'

    device: str = "cpu"

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.T <= 0:
            raise ValueError(f"T (terminal time) must be positive, got {self.T}")
        if self.N <= 0:
            raise ValueError(f"N (time steps) must be positive, got {self.N}")
        if self.num_paths <= 0:
            raise ValueError(f"num_paths must be positive, got {self.num_paths}")
        if self.picard_iterations <= 0:
            raise ValueError(f"picard_iterations must be positive, got {self.picard_iterations}")
        if self.global_iterations <= 0:
            raise ValueError(f"global_iterations must be positive, got {self.global_iterations}")
        if self.z_method not in ['gradient', 'regression']:
            raise ValueError(f"z_method must be 'gradient' or 'regression', got {self.z_method}")

    @property
    def dt(self) -> float:
        return self.T / self.N


@dataclass
class DGMConfig:
    """
    Hyperparameters for the Deep Galerkin Method solver.
    """
    T: float = 1.0
    dim_x: int = 1

    # Network
    n_layers: int = 3
    layer_width: int = 50

    # Sampling
    domain: Tuple[float, float] = (-2.0, 5.0)  # (lo, hi) bounds for collocation points
    N1: int = 1000          # interior batch size
    N2: int = 1000          # terminal batch size

    # Training
    n_stages: int = 10_000
    n_steps: int = 10
    learning_rate: float = 1e-4
    log_every: int = 1000       # stages between log messages
    verbose: bool = True

    device: str = "cpu"

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.T <= 0:
            raise ValueError(f"T (terminal time) must be positive, got {self.T}")
        if self.dim_x <= 0:
            raise ValueError(f"dim_x must be positive, got {self.dim_x}")
        if self.n_layers <= 0:
            raise ValueError(f"n_layers must be positive, got {self.n_layers}")
        if self.layer_width <= 0:
            raise ValueError(f"layer_width must be positive, got {self.layer_width}")
        if len(self.domain) != 2 or self.domain[0] >= self.domain[1]:
            raise ValueError(f"domain must be a (lo, hi) tuple with lo < hi, got {self.domain}")
        if self.N1 <= 0:
            raise ValueError(f"N1 must be positive, got {self.N1}")
        if self.N2 <= 0:
            raise ValueError(f"N2 must be positive, got {self.N2}")
        if self.n_stages <= 0:
            raise ValueError(f"n_stages must be positive, got {self.n_stages}")
        if self.n_steps <= 0:
            raise ValueError(f"n_steps must be positive, got {self.n_steps}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        if self.log_every <= 0:
            raise ValueError(f"log_every must be positive, got {self.log_every}")

    @property
    def total_steps(self) -> int:
        return self.n_stages * self.n_steps
# Deep Iterative Method for High-Dimensional FBSDEs

This repository contains the PyTorch implementation of the **Deep Iterative Method**, a numerical framework for solving high-dimensional Forward-Backward Stochastic Differential Equations (FBSDEs). This code accompanies my Master's thesis **["Deep Learning for High-Dimensional Forward-Backward Stochastic Differential Equations"](https://utoronto.scholaris.ca/items/15db450b-5204-4a59-8305-98f5e58e04da)** (University of Toronto, 2025).

FBSDEs provide the mathematical foundation for a wide range of problems in quantitative finance and engineering. They arise naturally in determining optimal strategies in dynamic, uncertain environments—such as derivative pricing, portfolio optimization, and stochastic control.

## Overview

This package solves general FBSDE systems of the form:

$$
\begin{cases}
dX_t = \mu(t, X_t, Y_t, Z_t) dt + \sigma(t, X_t, Y_t, Z_t) dW_t, & X_0 = x_0 \\
-dY_t = f(t, X_t, Y_t, Z_t) dt - Z_t dW_t, & Y_T = g(X_T)
\end{cases}
$$

where $X_t \in \mathbb{R}^d$ is the forward process, $Y_t \in \mathbb{R}^m$ is the backward process, and $Z_t \in \mathbb{R}^{m \times d}$ is the control process.

The solver overcomes the curse of dimensionality by approximating the solution maps $(t, x) \mapsto Y_t$ and $(t, x) \mapsto Z_t$ using deep neural networks. The architecture relies on a **Deep Picard Iteration**, effectively treating the solution as the fixed point of a contraction mapping on the space of stochastic processes.

## Key Capabilities

*   **Hierarchical Solver Support:**
    *   **Uncoupled:** Standard systems where forward dynamics are independent of $(Y, Z)$.
    *   **Coupled:** Systems where $X_t$ depends on $Y_t$ and $Z_t$, resolved via a Global Picard Iteration.
    *   **McKean-Vlasov:** Mean-field systems where coefficients depend on the law $\mathcal{L}(X_t, Y_t, Z_t)$.
*   **Rigorous Benchmarking:** Includes five benchmark equations with analytical solutions for the different problem classes. 
*   **Two Z-Approximation Schemes:** Supports both **Gradient-based** and **Regression-based** approximation for the control process.
*   **GPU Accelerated:** Fully vectorized PyTorch implementation supporting CUDA execution for high-dimensional problems.



## Installation

### From Source

```bash
# Clone the repository
git clone https://github.com/bilalhusain126/DIM-FBSDEs.git
cd DIM-FBSDEs

# Install the package
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

### Requirements

- Python ≥ 3.8
- PyTorch ≥ 2.0.0
- NumPy ≥ 1.24.0
- Matplotlib ≥ 3.7.0

## Quick Start

### Example: Solving Black-Scholes-Barenblatt Equation

The following example solves the 3D **Black-Scholes-Barenblatt** equation using the Deep Picard Iteration method.

```python
import torch
from dim_fbsde.equations import BSBEquation
from dim_fbsde.solvers import UncoupledFBSDESolver
from dim_fbsde.nets import MLP
from dim_fbsde.config import SolverConfig, TrainingConfig

# Configure device
device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Define system dynamics 
dim_x = 3
equation = BSBEquation(dim_x=dim_x, r=0.05, sigma=0.4, device=device)

# 2. Configure solver (shared settings)
base_solver_cfg = {
    'T': 1.0,
    'N': 120,
    'num_paths': 8000,
    'picard_iterations': 10,
    'device': device
}

# 3. Configure training
train_cfg = TrainingConfig(
    batch_size=500,
    epochs=5,
    learning_rate=1e-4,
    verbose=True
)

# 4. Initialize function approximators
input_dim = 1 + dim_x
hidden_layers = [64, 64, 64]

# 5. Solve with Gradient Method
print("Running Gradient Method...")
solver_cfg_grad = SolverConfig(**base_solver_cfg, z_method='gradient')
nn_Y_grad = MLP(input_dim=input_dim, output_dim=1, hidden_dims=hidden_layers)
solver_grad = UncoupledFBSDESolver(equation, solver_cfg_grad, train_cfg, nn_Y_grad, nn_Z=None)
solution_grad = solver_grad.solve()

# 6. Solve with Regression Method
print("Running Regression Method...")
solver_cfg_reg = SolverConfig(**base_solver_cfg, z_method='regression')
nn_Y_reg = MLP(input_dim=input_dim, output_dim=1, hidden_dims=hidden_layers)
nn_Z_reg = MLP(input_dim=input_dim, output_dim=dim_x, hidden_dims=hidden_layers)
solver_reg = UncoupledFBSDESolver(equation, solver_cfg_reg, train_cfg, nn_Y_reg, nn_Z_reg)
solution_reg = solver_reg.solve()

print("Both methods completed successfully")
```

### Visualization

```python
from dim_fbsde.utils import plot_pathwise_comparison

# Compare both methods against analytical solution
# Model-based evaluation ensures fair comparison on identical X paths
fig, axes = plot_pathwise_comparison(
    solutions=[solution_grad],                           # X paths from gradient method
    labels=['DIM (Gradient)', 'DIM (Regression)'],       # Method labels
    models=[
        (solver_grad.nn_Y, None),                        # Gradient: only Y network
        (solver_reg.nn_Y, solver_reg.nn_Z)               # Regression: Y and Z networks
    ],
    z_methods=['gradient', 'regression'],                # Z computation methods
    equation=equation,                                   # Needed for gradient Z = σ·∇Y
    analytical_Y_func=equation.analytical_y,             # True solution for comparison
    analytical_Z_func=equation.analytical_z,
    analytical_Y_kwargs={'T_terminal': solver_cfg_grad.T},
    analytical_Z_kwargs={'T_terminal': solver_cfg_grad.T},
    component_idx=0,                                     # Which Z component to plot
    device=device,
    num_paths_to_plot=5                                  # Number of sample paths
)
```

## Package Structure

```
dim_fbsde/
├── equations/             # FBSDE problem definitions
│   ├── base.py            # Abstract FBSDE interface
│   └── benchmarks.py      # Benchmark systems with analytical solutions
├── solvers/               # Numerical solvers
│   ├── uncoupled.py       # Deep Picard iteration for uncoupled systems
│   ├── coupled.py         # Global iteration for coupled systems
│   └── mckean_vlasov.py   # Global iteration for mean-field systems
├── nets/                  # Neural network architectures
│   └── mlp.py             # Multi-layer perceptron for Y and Z approximation
├── utils/                 # Utilities
│   └── visualizations.py  # Pathwise comparison and error analysis plots
└── config.py              # Configuration dataclasses (SolverConfig, TrainingConfig)
```

## Advanced Usage

### Custom Equations

You can define custom FBSDE systems by subclassing the `FBSDE` base class and implementing the required methods. Once defined, your custom equation can be used with any of the solvers in the library.

```python
from dim_fbsde.equations.base import FBSDE
import torch

class MyCustomEquation(FBSDE):
    def __init__(self, dim_x, **kwargs):
        super().__init__(dim_x=dim_x, dim_y=1, dim_w=dim_x,
                         x0=torch.ones(dim_x), **kwargs)

    def drift(self, t, x, y, z, **kwargs):
        # Define your forward drift
        return torch.zeros_like(x)

    def diffusion(self, t, x, y, z, **kwargs):
        # Define your forward diffusion
        return torch.diag_embed(x)

    def driver(self, t, x, y, z, **kwargs):
        # Define your backward driver
        return -y

    def terminal_condition(self, x, **kwargs):
        # Define terminal condition
        return torch.sum(x**2, dim=1, keepdim=True)
```

## Algorithm Details

### Deep Picard Iteration (Uncoupled Solver)

For uncoupled systems, the solver computes the fixed point of the solution map defined by the integral form of the BSDE. At each iteration k:

1. **Simulation**: Generate forward paths using Euler-Maruyama scheme
2. **Approximation**: Train neural network to minimize L²-error:

```math
\mathcal{L}(\theta) = \mathbb{E}\left[ \left| \mathcal{N}_Y(t_i, X_{t_i}) - \left( g(X_T) + \int_{t_i}^T f(s, X_s, Y_s^{(k-1)}, Z_s^{(k-1)}) ds \right) \right|^2 \right]
```

### Global Iteration (Coupled & Mean-Field)

For **Coupled** and **McKean-Vlasov** systems, a global fixed-point iteration resolves circular dependencies:

1. **Forward Step**: Simulate state process using fixed coefficients from previous estimates Y<sup>(k-1)</sup>, Z<sup>(k-1)</sup> (and empirical law for mean-field)
2. **Backward Step**: Solve the resulting uncoupled BSDE using Deep Picard Iteration to update Y<sup>(k)</sup>, Z<sup>(k)</sup>

### Z Estimation Schemes

The control process Z<sub>t</sub> is approximated using one of two methods:

1. **Gradient-Based** (`z_method='gradient'`): Computes Z<sub>t</sub> via automatic differentiation using the Feynman-Kac representation:
```math
Z_t = \nabla_x \mathcal{N}_Y(t, X_t) \cdot \sigma(t, X_t, Y_t, Z_t)
```

<br>

2. **Regression-Based** (`z_method='regression'`): Trains a secondary network to approximate the martingale representation term:
```math
Z_t \approx \frac{1}{\Delta t} \mathbb{E}\left[ (Y_{t+\Delta t} - Y_t) \Delta W_t^\top \mid \mathcal{F}_t \right]
```

## Examples

See the `notebooks/` directory for comprehensive demonstrations:
- `benchmark_demonstrations.ipynb`: Complete examples for the five benchmark problems with visualization and error analysis

## Citation

If you use this code in your research, please cite:

```bibtex
@mastersthesis{husain2025fbsde,
  author  = {Bilal Saleh Husain},
  title   = {Deep Learning for High-Dimensional Forward-Backward Stochastic Differential Equations},
  school  = {University of Toronto},
  year    = {2025},
  url     = {https://utoronto.scholaris.ca/items/15db450b-5204-4a59-8305-98f5e58e04da}
}
```

## Author

**Bilal Saleh Husain**

Master of Mathematics, University of Toronto, 2025

Contact: bilal.husain@mail.utoronto.ca | GitHub: [@bilalhusain126](https://github.com/bilalhusain126)

---

**Version**: 1.0.0

"""
Deep Iterative Method for Coupled FBSDEs.

This module implements the solver for fully coupled Forward-Backward Stochastic 
Differential Equations where the forward process X depends on the backward processes Y and Z.
It uses a global iterative scheme to resolve the circular dependency between the 
forward and backward dynamics.

References:
- Thesis Chapter 6: Deep Iterative Method for Coupled FBSDEs
- Thesis Section 6.2: A Global Picard Iteration Scheme
"""

import torch
import numpy as np
import logging
from typing import Dict, Any, Optional, List, Tuple

from dim_fbsde.config import SolverConfig, TrainingConfig
from dim_fbsde.equations.base import FBSDE
from dim_fbsde.solvers.uncoupled import UncoupledFBSDESolver

logger = logging.getLogger(__name__)


class CoupledFBSDESolver:
    """
    Solves coupled FBSDEs using a Global Picard Iteration.

    In a coupled system, the coefficients of the forward SDE depend on the backward 
    processes Y and Z. This creates a circular dependency that precludes a simple 
    sequential solution.
    
    The solver resolves this by iterating between:
    1.  **Forward Pass**: Simulating the forward process X^{(k+1)} using the 
        approximations Y^{(k)} and Z^{(k)} from the previous iteration.
    2.  **Backward Pass**: Solving for Y^{(k+1)} and Z^{(k+1)} by treating the 
        newly generated X^{(k+1)} as fixed inputs to an uncoupled BSDE solver.
    
    See Thesis Algorithm 4 (Section 6.3) for the complete procedure.
    """

    def __init__(self,
                 equation: FBSDE,
                 solver_config: SolverConfig,
                 training_config: TrainingConfig,
                 nn_Y: torch.nn.Module,
                 nn_Z: Optional[torch.nn.Module] = None,
                 forward_scheme: str = 'feedback'):
        """
        Initializes the Coupled Solver.

        Args:
            equation (FBSDE): The Coupled FBSDE system definition.
            solver_config (SolverConfig): Global hyperparameters (e.g., global_iterations).
            training_config (TrainingConfig): Training settings for the inner uncoupled solver.
            nn_Y (nn.Module): Neural network for approximating Y.
            nn_Z (nn.Module, optional): Neural network for approximating Z.
            forward_scheme (str): How the coupling terms are evaluated during the
                forward simulation.
                'feedback' (default): evaluate the current networks at the
                simulated state, so the drift and diffusion receive
                Y = N_Y(t, X_t) as a function of the state being simulated.
                This implements the Markovian iteration of Bender and Zhang
                (2008).
                'legacy': reuse the stored backward paths from the previous
                global iteration, matched by path index (Thesis Algorithm 4).
        """
        self.forward_scheme = forward_scheme
        self.eqn = equation
        self.cfg = solver_config
        self.train_cfg = training_config
        
        self.nn_Y = nn_Y
        self.nn_Z = nn_Z
        self.device = torch.device(self.cfg.device)

        # Internal Uncoupled Solver acts as the "Backward Step" engine.
        # It manages the neural network training for Y and Z.
        self.inner_solver = UncoupledFBSDESolver(
            equation=equation,
            solver_config=solver_config,
            training_config=training_config,
            nn_Y=nn_Y,
            nn_Z=nn_Z
        )

        # Global path storage (updated at each global iteration)
        self.X: Optional[torch.Tensor] = None
        self.Y_path: Optional[torch.Tensor] = None
        self.Z_path: Optional[torch.Tensor] = None
        self.global_history: List[Dict[str, float]] = []

    def solve(self) -> Dict[str, Any]:
        """
        Executes the Global Picard Iteration.

        Returns:
            Dict[str, Any]: A dictionary containing the final converged solution paths 
                            and the history of the global convergence error.
        """
        if self.train_cfg.verbose:
            logger.info(f"Starting Coupled Solver (Global Iters: {self.cfg.global_iterations}, Inner Iters: {self.cfg.picard_iterations})...")

        # 1. Initialization (k=0)
        # Initialize backward paths to zero as the starting guess (Thesis Section 6.2.1).
        M, N = self.cfg.num_paths, self.cfg.N
        self.Y_path = torch.zeros(M, N + 1, self.eqn.dim_y, device=self.device)
        self.Z_path = torch.zeros(M, N, self.eqn.dim_y, self.eqn.dim_w, device=self.device)

        # 2. Global Loop (Outer Iterations)
        for k in range(self.cfg.global_iterations):
            Y_prev_global = self.Y_path.clone()

            if self.train_cfg.verbose:
                logger.info(f"\n--- Global Iteration {k+1}/{self.cfg.global_iterations} ---")

            # Step A: Simulate Forward Process X^{(k+1)}
            # The drift and diffusion are evaluated using Y^{(k)} and Z^{(k)}.
            X_new, dW_new = self._simulate_coupled_forward()
            
            # Step B: Solve Backward Process (Inner Solver)
            # We inject the specific X paths we just generated into the inner solver,
            # effectively reducing the problem to an Uncoupled BSDE conditioned on X^{(k+1)}.
            # See Thesis Section 6.2.3.
            self.inner_solver.set_external_paths(X_new.cpu().numpy(), dW_new.cpu().numpy())
            
            # Run the inner Deep Picard solver
            # This trains the neural networks further on the new forward distribution.
            solution_inner = self.inner_solver.solve()
            
            # Update global paths with the new approximations
            self.X = torch.tensor(solution_inner['X'], device=self.device)
            self.Y_path = torch.tensor(solution_inner['Y'], device=self.device)
            self.Z_path = torch.tensor(solution_inner['Z'], device=self.device)

            # Step C: Convergence Check
            # Monitor the RMS change in the Y trajectories between global iterations.
            diff = self.Y_path - Y_prev_global
            global_error = torch.sqrt(torch.mean(diff**2)).item()
            
            if self.train_cfg.verbose:
                logger.info(f"Global Error (Y_k vs Y_k-1): {global_error:.4e}")
            
            self.global_history.append({'iteration': k+1, 'global_error': global_error})

        return self.get_results()

    def _simulate_coupled_forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Simulates the forward process X with coupled drift and diffusion coefficients.

        The coupling terms are evaluated according to the configured forward scheme.
        'feedback' evaluates the current networks at the simulated state, so each
        Euler-Maruyama step receives Y = N_Y(t, X_t) at the point the path is
        currently at. 'legacy' reuses the stored backward paths from the previous
        global iteration, matched by path index; because each global iteration
        draws fresh Brownian increments, these stored values correspond to states
        from the previous iteration rather than the state being simulated.

        See Thesis Equation 6.2.1.
        """
        M = self.cfg.num_paths
        N = self.cfg.N
        dt = self.cfg.dt

        t_grid = torch.linspace(0, self.cfg.T, N + 1, device=self.device)

        X = torch.zeros(M, N + 1, self.eqn.dim_x, device=self.device)
        dW = torch.randn(M, N, self.eqn.dim_w, device=self.device) * np.sqrt(dt)

        # Initial condition
        # self.eqn.x0 already has shape [1, dim_x], just repeat to [M, dim_x]
        X[:, 0, :] = self.eqn.x0.repeat(M, 1)

        use_feedback = self.forward_scheme == 'feedback'
        if use_feedback:
            self.nn_Y.eval()
            if self.nn_Z is not None:
                self.nn_Z.eval()

        for i in range(N):
            t = t_grid[i]
            x_curr = X[:, i, :]

            if use_feedback:
                # Evaluate the current decoupling-field approximations at the
                # simulated state. Without a Z-network (gradient scheme), Z has
                # no state-function representation here, so the stored paths
                # remain the only available estimate for z_curr.
                nn_in = torch.cat([t.repeat(M).unsqueeze(1), x_curr], dim=1)
                with torch.no_grad():
                    y_curr = self.nn_Y(nn_in)
                    if self.nn_Z is not None:
                        z_curr = self.nn_Z(nn_in).reshape(M, self.eqn.dim_y, self.eqn.dim_w)
                    else:
                        z_curr = self.Z_path[:, i, :, :]
            else:
                # Retrieve the estimates Y^{(k)}_t and Z^{(k)}_t at the current time step i
                y_curr = self.Y_path[:, i, :]
                z_curr = self.Z_path[:, i, :, :]

            # Evaluate coefficients with coupling
            # drift = mu(t, X_t, Y_t, Z_t)
            drift = self.eqn.drift(t, x_curr, y_curr, z_curr, T_terminal=self.cfg.T)
            # sigma = sigma(t, X_t, Y_t, Z_t)
            sigma = self.eqn.diffusion(t, x_curr, y_curr, z_curr, T_terminal=self.cfg.T)

            # Euler-Maruyama Update
            diffusion_term = torch.bmm(sigma, dW[:, i, :].unsqueeze(-1)).squeeze(-1)
            X[:, i+1, :] = x_curr + drift * dt + diffusion_term

        return X, dW

    def get_results(self) -> Dict[str, Any]:
        """Returns the final simulation results."""
        return {
            'time': np.linspace(0, self.cfg.T, self.cfg.N + 1),
            'X': self.X.cpu().numpy(),
            'Y': self.Y_path.cpu().numpy(),
            'Z': self.Z_path.cpu().numpy(),
            'global_history': self.global_history
        }
        
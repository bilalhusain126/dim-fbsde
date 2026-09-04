"""
Deep Iterative Method for McKean-Vlasov FBSDEs.

This module implements the solver for McKean-Vlasov (Mean-Field) FBSDEs, where the 
coefficients depend on the probability distribution (law) of the solution processes.
The algorithm resolves the dependency on the law via a global fixed-point iteration 
over the empirical distribution of simulated paths.

References:
- Thesis Chapter 7: Deep Iterative Method for McKean-Vlasov FBSDEs
- Thesis Section 7.2: A Global Picard Iteration Scheme
"""

import torch
import numpy as np
import logging
from typing import Dict, Any, Optional, List, Tuple

from dim_fbsde.config import SolverConfig, TrainingConfig
from dim_fbsde.equations.base import FBSDE
from dim_fbsde.solvers.uncoupled import UncoupledFBSDESolver

logger = logging.getLogger(__name__)


class McKeanVlasovSolver:
    """
    Solves McKean-Vlasov (Mean-Field) FBSDEs using a Global Picard Iteration.

    McKean-Vlasov equations involve coefficients that depend on the law of the process 
    (e.g., E[X_t], Var(X_t)). The solver handles this by:
    1.  Using empirical statistics computed from the entire batch of simulated paths 
        to approximate the law.
    2.  Iterating globally to find a fixed point for both the process paths and 
        their empirical distribution.
    
    The Global Loop proceeds as follows (Thesis Algorithm 5):
    1.  **Compute Laws**: Calculate statistics (means/distributions) from the 
        previous iteration's paths (X, Y, Z).
    2.  **Forward Pass**: Simulate X^{(k+1)} using drift/diffusion coefficients 
        evaluated with these "frozen" statistics.
    3.  **Backward Pass**: Solve for Y^{(k+1)}, Z^{(k+1)} using the inner Deep Picard 
        solver, injecting the frozen statistics into the driver and terminal conditions.
    """

    def __init__(self,
                 equation: FBSDE,
                 solver_config: SolverConfig,
                 training_config: TrainingConfig,
                 nn_Y: torch.nn.Module,
                 nn_Z: Optional[torch.nn.Module] = None):
        """
        Initializes the McKean-Vlasov Solver.

        Args:
            equation (FBSDE): The McKean-Vlasov FBSDE system.
            solver_config (SolverConfig): Global hyperparameters (e.g., global_iterations).
            training_config (TrainingConfig): Training settings for the inner uncoupled solver.
            nn_Y (nn.Module): Neural network for approximating Y.
            nn_Z (nn.Module, optional): Neural network for approximating Z.
        """
        self.eqn = equation
        self.cfg = solver_config
        self.train_cfg = training_config
        
        self.nn_Y = nn_Y
        self.nn_Z = nn_Z
        self.device = torch.device(self.cfg.device)

        # Global path storage (updated at each global iteration)
        self.X: Optional[torch.Tensor] = None
        self.Y_path: Optional[torch.Tensor] = None
        self.Z_path: Optional[torch.Tensor] = None
        self.global_history: List[Dict[str, float]] = []

    def solve(self) -> Dict[str, Any]:
        """
        Executes the Global Picard Iteration for MV-FBSDEs.

        Returns:
            Dict[str, Any]: A dictionary containing the final converged solution paths 
                            and the history of the global convergence error.
        """
        if self.train_cfg.verbose:
            logger.info(f"Starting McKean-Vlasov Solver (Global Iters: {self.cfg.global_iterations}, Inner Iters: {self.cfg.picard_iterations})...")

        # 1. Initialization (k=0)
        M, N = self.cfg.num_paths, self.cfg.N
        self.Y_path = torch.zeros(M, N + 1, self.eqn.dim_y, device=self.device)
        self.Z_path = torch.zeros(M, N, self.eqn.dim_y, self.eqn.dim_w, device=self.device)
        
        # Initial X simulation: Assume zero interaction/mean initially.
        # This provides the initial distribution estimate L^(0).
        self.X, _ = self._simulate_mv_forward(is_initial=True)

        # 2. Global Loop (Outer Iterations)
        for k in range(self.cfg.global_iterations):
            Y_prev_global = self.Y_path.clone()

            if self.train_cfg.verbose:
                logger.info(f"\n--- Global Iteration {k+1}/{self.cfg.global_iterations} ---")

            # Step A: Forward Pass (Simulate X)
            # Simulate X_{k+1} using drift/diffusion conditioned on statistics from (X_k, Y_k).
            # The _simulate_mv_forward method handles the statistics calculation internally.
            X_new, dW_new = self._simulate_mv_forward(is_initial=False)
            
            # Step B: Backward Pass (Inner Solver)
            # We instantiate a fresh Uncoupled Solver for this step to keep state clean.
            inner_solver = UncoupledFBSDESolver(
                equation=self.eqn,
                solver_config=self.cfg,
                training_config=self.train_cfg,
                nn_Y=self.nn_Y,
                nn_Z=self.nn_Z
            )
            
            # Inject the newly simulated X paths into the inner solver.
            inner_solver.set_external_paths(X_new.cpu().numpy(), dW_new.cpu().numpy())
            
            # --- Runtime Proxy Setup (Crucial for Law Dependence) ---
            # The inner solver works on standard BSDEs (f(t,x,y,z)).
            # MV-BSDEs require f(t,x,y,z, Law(X)).
            # We use "Proxy Methods" to inject the empirical law into the equation calls.
            
            # 1. Capture the ORIGINAL bound methods first (to avoid recursion).
            original_driver = self.eqn.driver
            original_terminal = self.eqn.terminal_condition

            # 2. Define Proxies using the CAPTURED methods.
            def driver_with_law_proxy(t, x, y, z, **kwargs):
                # Calculate time index to grab the correct slice of the distribution
                idx = int(round(t.item() / self.cfg.dt))
                idx = min(idx, self.cfg.N)
                
                # Full distribution from PREVIOUS iteration (standard Picard scheme)
                dist_x = self.X[:, idx, :] 
                dist_y = self.Y_path[:, idx, :]
                
                # Call the CAPTURED original method with the injected distribution
                return original_driver(t, x, y, z, mean_x=dist_x, mean_y=dist_y, **kwargs)

            def terminal_with_law_proxy(x, **kwargs):
                # Full distribution at terminal time from NEW paths
                dist_x_T = X_new[:, -1, :]
                return original_terminal(x, mean_x=dist_x_T, **kwargs)

            # 3. Apply the Monkey-Patch to the equation instance shared with inner_solver
            inner_solver.eqn.driver = driver_with_law_proxy
            inner_solver.eqn.terminal_condition = terminal_with_law_proxy

            try:
                # Solve the uncoupled BSDE with the "frozen" law coefficients
                solution_inner = inner_solver.solve()
            finally:
                # 4. Cleanup: Restore original methods immediately after solve
                # This prevents state corruption or recursion in the next iteration.
                inner_solver.eqn.driver = original_driver
                inner_solver.eqn.terminal_condition = original_terminal

            # Step C: Update Global Iterates
            self.X = torch.tensor(solution_inner['X'], device=self.device)
            self.Y_path = torch.tensor(solution_inner['Y'], device=self.device)
            self.Z_path = torch.tensor(solution_inner['Z'], device=self.device)

            # Step D: Convergence Check
            # Monitor the RMS change in the Y trajectories between global iterations.
            diff = self.Y_path - Y_prev_global
            global_error = torch.sqrt(torch.mean(diff**2)).item()
            
            if self.train_cfg.verbose:
                logger.info(f"Global Error (Y_k vs Y_k-1): {global_error:.4e}")
            
            self.global_history.append({'iteration': k+1, 'global_error': global_error})

        return self.get_results()

    def _simulate_mv_forward(self, is_initial: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Simulates the forward process X.
        
        If `is_initial` is False, it calculates the drift and diffusion coefficients using the 
        empirical statistics (law) derived from the stored global paths (self.X, self.Y_path).
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

        for i in range(N):
            t = t_grid[i]
            x_curr = X[:, i, :]
            
            mean_x = None
            mean_y = None
            
            # Placeholders for initial run
            y_curr = torch.zeros(M, self.eqn.dim_y, device=self.device)
            z_curr = torch.zeros(M, self.eqn.dim_y, self.eqn.dim_w, device=self.device)

            if not is_initial:
                # Use stored paths from previous iteration
                y_curr = self.Y_path[:, i, :]
                z_curr = self.Z_path[:, i, :, :]
                
                # Pass the full distribution (batch) to the equation
                # The equation will handle the pairwise distance / kernel logic
                mean_x = self.X[:, i, :] 
                mean_y = self.Y_path[:, i, :]

            # 'mean_x' argument carries the full batch (or None)
            drift = self.eqn.drift(t, x_curr, y_curr, z_curr, 
                                   mean_x=mean_x, mean_y=mean_y, 
                                   T_terminal=self.cfg.T)
                                   
            sigma = self.eqn.diffusion(t, x_curr, y_curr, z_curr, 
                                       mean_x=mean_x, mean_y=mean_y, 
                                       T_terminal=self.cfg.T)
            
            # Euler-Maruyama Step
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
        
"""
Deep Picard Iteration Solver for Uncoupled FBSDEs.

This module implements the core solver for uncoupled Forward-Backward Stochastic 
Differential Equations using the Deep Picard Iteration method. It handles the 
simulation of the forward process and the iterative approximation of the backward 
processes (Y, Z) via neural networks.

References:
- Thesis Chapter 5: Deep Picard Iteration for Uncoupled FBSDEs
- Thesis Chapter 4: Numerical Approximation of Conditional Expectations
"""

import torch
import torch.nn as nn
import numpy as np
import copy
import logging
from typing import Dict, Any, Tuple, Optional, List

from dim_fbsde.config import SolverConfig, TrainingConfig
from dim_fbsde.equations.base import FBSDE

logger = logging.getLogger(__name__)


class UncoupledFBSDESolver:
    """
    Solves uncoupled FBSDEs using the Deep Picard Iteration method.

    The algorithm relies on the nonlinear Feynman-Kac formula and fixed-point iterations.
    It proceeds in two main phases:
    1.  **Forward Simulation**: The forward process X is simulated using the Euler-Maruyama scheme. 
        Since the system is uncoupled, X is independent of Y and Z.
    2.  **Backward Iteration (Deep Picard)**: The solution functions Y(t,x) and Z(t,x) are approximated 
        via neural networks. In each Picard iteration, the networks are trained to minimize the mean 
        squared error against a target derived from the integral form of the BSDE.
    
    See Thesis Section 5.2 for the mathematical derivation of the Picard scheme.
    """

    def __init__(self,
                 equation: FBSDE,
                 solver_config: SolverConfig,
                 training_config: TrainingConfig,
                 nn_Y: nn.Module,
                 nn_Z: Optional[nn.Module] = None):
        """
        Initializes the solver.

        Args:
            equation (FBSDE): The physics definition of the problem (drift, diffusion, driver, terminal).
            solver_config (SolverConfig): Numerical hyperparameters (time horizon, discretization, etc.).
            training_config (TrainingConfig): Optimization hyperparameters (learning rate, batch size, etc.).
            nn_Y (nn.Module): Neural network instance for approximating the value process Y(t, x).
            nn_Z (nn.Module, optional): Neural network instance for approximating the control process Z(t, x).
                                        Required only if z_method is 'regression'.
        """
        self.eqn = equation
        self.cfg = solver_config
        self.train_cfg = training_config
        
        self.nn_Y = nn_Y
        self.nn_Z = nn_Z

        # Validate configuration for Z estimation method
        if self.cfg.z_method == 'regression' and self.nn_Z is None:
            raise ValueError("nn_Z must be provided when z_method is 'regression'.")

        # Device management (CPU/GPU)
        self.device = torch.device(self.cfg.device)
        self.nn_Y.to(self.device)
        if self.nn_Z:
            self.nn_Z.to(self.device)

        # Placeholders for simulation data (tensors on device)
        self.time_grid: Optional[torch.Tensor] = None
        self.X: Optional[torch.Tensor] = None         # Forward state: [M, N+1, dx]
        self.dW: Optional[torch.Tensor] = None        # Brownian increments: [M, N, dw]
        
        # Placeholders for solution iterates
        self.Y: Optional[torch.Tensor] = None         # Value process: [M, N+1, dy]
        self.Z: Optional[torch.Tensor] = None         # Control process: [M, N, dy, dw]
        
        self.history: List[Dict[str, Any]] = []

    def solve(self) -> Dict[str, Any]:
        """
        Executes the Deep Picard Iteration loop to solve the FBSDE.

        See Thesis Section 5.4 (Algorithm Summary) for the complete procedure.

        Returns:
            Dict[str, Any]: A dictionary containing the solution paths ('X', 'Y', 'Z'), 
                            the time grid, and the convergence history.
        """
        if self.train_cfg.verbose:
            logger.info(f"Starting Uncoupled Solver ({self.cfg.z_method})...")
        
        # 1. Simulate Forward Process
        # If paths were not injected externally (e.g., from a coupled wrapper), simulate them now.
        if self.X is None:
            self._simulate_forward_process()

        # 2. Initialization
        # Initialize Y and Z paths to zero before the first iteration.
        self.Y = torch.zeros(self.cfg.num_paths, self.cfg.N + 1, self.eqn.dim_y, device=self.device)
        self.Z = None if self.cfg.z_method == 'none' else \
            torch.zeros(self.cfg.num_paths, self.cfg.N, self.eqn.dim_y, self.eqn.dim_w, device=self.device)
        
        # Enforce terminal condition Y_T = g(X_T) explicitly.
        self.Y[:, -1, :] = self.eqn.terminal_condition(self.X[:, -1, :])

        # 3. Picard Iteration Loop
        # Perform an initial update to populate Y and Z based on the random initialization of the networks.
        self.Y, self.Z = self._update_iterates(self.Y, self.Z) 
        
        # Main Fixed-Point Iteration
        for k in range(self.cfg.picard_iterations):
            Y_prev = self.Y.clone()
            Z_prev = None if self.Z is None else self.Z.clone()

            # Step A: Optimization (Training Phase)
            # Train the networks to approximate the conditional expectations for the next iterate.
            # See Thesis Section 4.2 regarding "Approximation as a Supervised Learning Problem".
            y_loss, z_loss = self._train_networks(Y_prev, Z_prev)
            
            # Step B: Update (Evaluation Phase)
            # Generate new path estimates using the trained networks.
            self.Y, self.Z = self._update_iterates(Y_prev, Z_prev)
            
            # Step C: Convergence Monitoring
            # Compute RMSE between consecutive iterates (excluding terminal time which is fixed).
            error = torch.sqrt(torch.mean((self.Y[:, :-1, :] - Y_prev[:, :-1, :])**2)).item()
            
            if self.train_cfg.verbose:
                logger.info(f"Picard Iter {k+1}/{self.cfg.picard_iterations}: Error={error:.4e} | Mean Y_Loss={np.mean(y_loss):.4e}")

            # Record metrics for analysis
            record = {
                "iteration": k + 1,
                "error": error,
                "y_loss": y_loss,
                "z_loss": z_loss,
                "nn_Y_state": copy.deepcopy(self.nn_Y.state_dict())
            }
            self.history.append(record)

        return self.get_results()

    def _simulate_forward_process(self):
        """
        Simulates the forward process X using the Euler-Maruyama scheme.
        
        See Thesis Section 3.1: The Euler-Maruyama Method.
        This generates the training distribution for the subsequent neural network training.
        Brownian increments dW are stored for use in the 'regression' Z-method.
        """
        M = self.cfg.num_paths
        N = self.cfg.N
        dt = self.cfg.dt
        
        t_grid = torch.linspace(0, self.cfg.T, N + 1, device=self.device)
        self.time_grid = t_grid
        
        X = torch.zeros(M, N + 1, self.eqn.dim_x, device=self.device)
        dW = torch.randn(M, N, self.eqn.dim_w, device=self.device) * np.sqrt(dt)
        
        # Initialize X_0
        # self.eqn.x0 already has shape [1, dim_x], just repeat to [M, dim_x]
        X[:, 0, :] = self.eqn.x0.repeat(M, 1)
        
        # Create dummy placeholders for Y and Z (not used in uncoupled drift/diffusion)
        dummy_y = torch.zeros(M, self.eqn.dim_y, device=self.device)
        dummy_z = torch.zeros(M, self.eqn.dim_y, self.eqn.dim_w, device=self.device)

        # Time-stepping loop
        for i in range(N):
            t = t_grid[i]
            x_curr = X[:, i, :]
            
            drift = self.eqn.drift(t, x_curr, dummy_y, dummy_z, T_terminal=self.cfg.T)
            sigma = self.eqn.diffusion(t, x_curr, dummy_y, dummy_z, T_terminal=self.cfg.T)
            
            # Euler-Maruyama Update: X_{t+1} = X_t + mu*dt + sigma*dW
            # bmm handles batch matrix-vector multiplication: [M, dx, dw] @ [M, dw, 1] -> [M, dx, 1]
            diffusion_term = torch.bmm(sigma, dW[:, i, :].unsqueeze(-1)).squeeze(-1)
            X[:, i+1, :] = x_curr + drift * dt + diffusion_term
            
        self.X = X
        self.dW = dW

    def _compute_driver_sum(self, Y_tensor, Z_tensor):
        """
        Computes the backward cumulative Riemann sum of the driver function.
        
        Target: S_t = sum_{j=i}^{N-1} f(t_j, X_j, Y_j, Z_j) * dt
        This term approximates the integral in the BSDE fixed-point definition (Thesis Eq 5.2.1).
        """
        S = torch.zeros_like(Y_tensor, device=self.device)
        
        # Iterate backward in time to accumulate the sum
        for i in range(self.cfg.N - 1, -1, -1):
            t = self.time_grid[i]
            X_i = self.X[:, i, :]
            Y_i = Y_tensor[:, i, :]
            Z_i = None if Z_tensor is None else Z_tensor[:, i, :, :]
            
            f_val = self.eqn.driver(t, X_i, Y_i, Z_i, T_terminal=self.cfg.T)
            
            # Accumulate: Current driver value + sum from future steps
            S[:, i, :] = f_val * self.cfg.dt
            if i < self.cfg.N:
                S[:, i, :] += S[:, i+1, :]
        return S

    def _train_networks(self, Y_prev, Z_prev):
        """
        Prepares training data and executes the optimization loop for the neural networks.
        
        For Y: The target is Y_T + Integral(driver). See Thesis Eq 5.3.2.
        For Z (Regression): The target is derived from the martingale representation theorem.
                            See Thesis Eq 5.2.5.
        """
        # --- 1. Train Y Network ---
        S_cumulative = self._compute_driver_sum(Y_prev, Z_prev)
        terminal_Y = self.eqn.terminal_condition(self.X[:, -1, :], T_terminal=self.cfg.T)
        
        # Construct dataset for supervised learning
        inputs_list, targets_list = [], []
        for i in range(self.cfg.N):
            t_vec = self.time_grid[i].repeat(self.cfg.num_paths).unsqueeze(1)
            x_vec = self.X[:, i, :]
            
            # Input: (t, X_t)
            inputs_list.append(torch.cat([t_vec, x_vec], dim=1))
            # Target: g(X_T) + Sum_{j=i}^{T} f(...) * dt
            targets_list.append(terminal_Y + S_cumulative[:, i, :])
            
        all_inputs = torch.cat(inputs_list, dim=0)
        all_targets = torch.cat(targets_list, dim=0)
        
        y_losses = self._optimize_net(self.nn_Y, all_inputs, all_targets)
        
        # --- 2. Train Z Network (Regression Method Only) ---
        z_losses = []
        if self.cfg.z_method == 'regression' and self.nn_Z:
            # Re-evaluate Y using the *just trained* network to get cleaner gradients/differences
            with torch.no_grad():
                # Correctly reshape the flat network output back to [Time, Path, Dim]
                Y_updated_flat = self.nn_Y(all_inputs)
                Y_updated_time_first = Y_updated_flat.reshape(self.cfg.N, self.cfg.num_paths, -1)
                
                # Permute to [Path, Time, Dim] for consistency
                Y_updated = Y_updated_time_first.permute(1, 0, 2)
                Y_updated = torch.cat([Y_updated, terminal_Y.unsqueeze(1)], dim=1)
                
            inputs_z, targets_z = [], []
            dY = Y_updated[:, 1:, :] - Y_updated[:, :-1, :] # Forward difference of Y
            
            for i in range(self.cfg.N):
                t_vec = self.time_grid[i].repeat(self.cfg.num_paths).unsqueeze(1)
                x_vec = self.X[:, i, :]
                inputs_z.append(torch.cat([t_vec, x_vec], dim=1))
                
                # Z target approx: E[(Y_{t+1} - Y_t) * dW_t] / dt (Thesis Eq 5.3.5)
                # Batch matrix multiplication: [M, dy, 1] x [M, 1, dw] -> [M, dy, dw]
                target_i = torch.bmm(dY[:, i, :].unsqueeze(2), self.dW[:, i, :].unsqueeze(1)) / self.cfg.dt
                targets_z.append(target_i.reshape(self.cfg.num_paths, -1))
                
            all_inputs_z = torch.cat(inputs_z, dim=0)
            all_targets_z = torch.cat(targets_z, dim=0)
            
            z_losses = self._optimize_net(self.nn_Z, all_inputs_z, all_targets_z)
            
        return y_losses, z_losses

    def _optimize_net(self, model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor) -> List[float]:
        """
        Standard PyTorch optimization loop using Adam and StepLR scheduling.
        See Thesis Section 4.2.2 regarding Network Training.
        """
        optimizer = torch.optim.Adam(model.parameters(), lr=self.train_cfg.learning_rate)
        
        # Learning Rate Scheduler: Decays LR every 'step_size' optimization steps
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=self.train_cfg.lr_decay_step, 
            gamma=self.train_cfg.lr_decay_rate
        )
        
        loss_fn = nn.MSELoss()
        
        dataset_size = inputs.shape[0]
        indices = torch.arange(dataset_size)
        model.train()
        losses = []
        
        # Epoch Loop
        for _ in range(self.train_cfg.epochs):
            shuffled_idx = indices[torch.randperm(dataset_size)]
            
            # Mini-batch Loop
            for start_idx in range(0, dataset_size, self.train_cfg.batch_size):
                batch_idx = shuffled_idx[start_idx : start_idx + self.train_cfg.batch_size]
                batch_in = inputs[batch_idx]
                batch_target = targets[batch_idx]
                
                optimizer.zero_grad()
                pred = model(batch_in)
                loss = loss_fn(pred, batch_target)
                loss.backward()
                
                # Gradient Clipping to prevent instability
                if self.train_cfg.gradient_clip_val > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.train_cfg.gradient_clip_val)
                    
                optimizer.step()
                scheduler.step() # Step the scheduler after every batch update
                
                losses.append(loss.item())
        return losses

    def _update_iterates(self, Y_prev, Z_prev) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluates the trained networks to generate updated path estimates for Y and Z.
        
        For Z-Gradient method: Computes Z via Automatic Differentiation (AD).
        Z_t = nabla_x Y_t * sigma(t, X_t) (Thesis Eq 5.2.3)
        """
        self.nn_Y.eval()
        if self.nn_Z: self.nn_Z.eval()
        
        Y_new = torch.zeros_like(Y_prev)
        Z_new = None if Z_prev is None else torch.zeros_like(Z_prev)
        M = self.cfg.num_paths
        
        for i in range(self.cfg.N):
            t = self.time_grid[i]
            t_vec = t.repeat(M).unsqueeze(1)
            X_i = self.X[:, i, :]
            
            nn_in = torch.cat([t_vec, X_i], dim=1)
            
            # --- Update Y ---
            with torch.no_grad():
                Y_new[:, i, :] = self.nn_Y(nn_in)
            
            # --- Update Z ---
            if self.cfg.z_method == 'gradient':
                # Enable gradient tracking for X specifically to compute dY/dX
                X_i_grad = X_i.clone().requires_grad_(True)
                nn_input_grad = torch.cat([t_vec, X_i_grad], dim=1)
                Y_pred_grad = self.nn_Y(nn_input_grad)
                
                # Compute diffusion matrix sigma(t, X_t)
                with torch.no_grad():
                    # For uncoupled, Y_prev/Z_prev don't affect sigma, but passed for API consistency
                    sigma_i = self.eqn.diffusion(t, X_i, Y_prev[:,i,:], Z_prev[:,i,:,:], T_terminal=self.cfg.T)
                
                # Compute Jacobian J = dY/dX row-by-row
                jacobian_rows = []
                for j in range(self.eqn.dim_y):
                    grad_Yj = torch.autograd.grad(
                        outputs=Y_pred_grad[:, j].sum(),
                        inputs=X_i_grad,
                        create_graph=True, 
                        retain_graph=True
                    )[0]
                    jacobian_rows.append(grad_Yj.unsqueeze(1))
                
                if jacobian_rows:
                    jacobian_Y_X = torch.cat(jacobian_rows, dim=1) # Shape: [M, dy, dx]
                    # Z = Jacobian * Sigma
                    Z_new[:, i, :, :] = torch.bmm(jacobian_Y_X, sigma_i).detach()
                    
            elif self.cfg.z_method == 'regression':
                # Direct evaluation of the Z-network
                with torch.no_grad():
                    Z_pred_flat = self.nn_Z(nn_in)
                Z_new[:, i, :, :] = Z_pred_flat.reshape(M, self.eqn.dim_y, self.eqn.dim_w)
                
        # Set terminal value exactly
        Y_new[:, -1, :] = self.eqn.terminal_condition(self.X[:, -1, :], T_terminal=self.cfg.T)
        return Y_new, Z_new

    def get_results(self) -> Dict[str, Any]:
        """Returns the simulation results as numpy arrays for analysis."""
        return {
            'time': self.time_grid.cpu().numpy(),
            'X': self.X.cpu().numpy(),
            'Y': self.Y.cpu().numpy(),
            'Z': None if self.Z is None else self.Z.cpu().numpy(),
            'history': self.history
        }

    def set_external_paths(self, X_paths: np.ndarray, dW_paths: np.ndarray):
        """
        Injects pre-simulated paths.
        
        This is used by the Coupled and McKean-Vlasov solvers to pass forward paths
        that were generated based on previous backward iterates.
        """
        self.X = torch.tensor(X_paths, dtype=torch.float32, device=self.device)
        self.dW = torch.tensor(dW_paths, dtype=torch.float32, device=self.device)
        M, N_plus_1, _ = self.X.shape
        self.cfg.num_paths = M
        self.cfg.N = N_plus_1 - 1
        self.time_grid = torch.linspace(0, self.cfg.T, self.cfg.N + 1, device=self.device)
        
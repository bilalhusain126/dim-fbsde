"""
Deep Galerkin Method Solver for Uncoupled FBSDEs.

Solves the semilinear parabolic PDE derived from the nonlinear Feynman-Kac
theorem (Theorem 2.5.1). The PDE residual is constructed automatically from
the FBSDE coefficients — no hand-derived PDE is required.

References:
- Thesis Chapter 8: The Deep Galerkin Method
- Thesis Section 8.2: Application to Uncoupled FBSDEs
"""

import torch
import logging
from typing import Dict, Any, List

from dim_fbsde.config import DGMConfig
from dim_fbsde.equations.base import FBSDE
from dim_fbsde.nets.dgm import DGMNet

logger = logging.getLogger(__name__)


class DGMSolver:
    """
    Solves uncoupled FBSDEs via the Deep Galerkin Method.

    Given an FBSDE with coefficients (mu, sigma, f, F), the solver
    automatically constructs the Feynman-Kac PDE:

        du/dt + (grad u)^T mu + 0.5 Tr(sigma sigma^T H_x u)
              + f(t, x, u, J_x u * sigma) = 0

        u(T, x) = F(x)

    and trains a DGMNet to minimize the PDE residual + terminal condition loss.
    """

    def __init__(self, equation: FBSDE, config: DGMConfig):
        """
        Initializes the solver.

        Args:
            equation (FBSDE): The physics definition of the problem (drift, diffusion, driver, terminal).
            config (DGMConfig): Hyperparameters for the network, sampling, and training.
        """
        self.eqn = equation
        self.cfg = config
        self.device = torch.device(config.device)

        self.model = DGMNet(
            layer_width=config.layer_width,
            n_layers=config.n_layers,
            input_dim=config.dim_x,
            final_trans=None,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=config.learning_rate
        )

        self.history: List[float] = []

    def _compute_pde_residual(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the Feynman-Kac PDE residual at collocation points.

        Constructs the full PDE operator from the FBSDE coefficients:
            R = du/dt + (grad_x u)^T mu + 0.5 Tr(sigma sigma^T H_x u)
                + f(t, x, u, (J_x u) sigma)

        All derivatives are computed via automatic differentiation.
        """
        t.requires_grad_(True)
        x.requires_grad_(True)

        tx = torch.cat([t, x], dim=1)
        u = self.model(tx)  # [N1, 1]
        batch_size = t.shape[0]
        dim_x = self.cfg.dim_x

        # du/dt
        du_dt = torch.autograd.grad(
            u, t, grad_outputs=torch.ones_like(u),
            create_graph=True, retain_graph=True
        )[0]  # [N1, 1]

        # du/dx (spatial gradient)
        du_dx = torch.autograd.grad(
            u, x, grad_outputs=torch.ones_like(u),
            create_graph=True, retain_graph=True
        )[0]  # [N1, dim_x]

        # --- Evaluate FBSDE coefficients ---
        # For uncoupled systems, drift and diffusion don't depend on Y, Z,
        # but the API requires them. Pass dummy values.
        dummy_y = torch.zeros(batch_size, self.eqn.dim_y, device=self.device)
        dummy_z = torch.zeros(batch_size, self.eqn.dim_y, self.eqn.dim_w,
                              device=self.device)

        with torch.no_grad():
            mu = self.eqn.drift(t, x, dummy_y, dummy_z,
                                T_terminal=self.cfg.T)
            # mu: [N1, dim_x]

        # sigma: [N1, dim_x, dim_w]
        with torch.no_grad():
            sigma = self.eqn.diffusion(
                t, x, dummy_y, dummy_z, T_terminal=self.cfg.T
            )  # [N1, dim_x, dim_w]

        # --- Drift term: (grad_x u)^T mu ---
        # du_dx: [N1, dim_x], mu: [N1, dim_x]
        drift_term = (du_dx * mu).sum(dim=1, keepdim=True)  # [N1, 1]

        # --- Diffusion term: 0.5 Tr(sigma sigma^T H_x u) ---
        # sigma sigma^T: [N1, dim_x, dim_x]
        sigma_sigmaT = torch.bmm(sigma, sigma.transpose(1, 2))  # [N1, dx, dx]

        # Compute Hessian diagonal entries weighted by sigma_sigmaT.
        # For efficiency, compute sum_i sum_j (sigma_sigmaT)_{ij} d2u/dxi dxj.
        # This equals Tr(sigma sigma^T H_x u).
        #
        # We compute d2u/dxi dxj by differentiating du/dxi w.r.t. x, then
        # dot with sigma_sigmaT row i.
        trace_term = torch.zeros(batch_size, 1, device=self.device)
        for i in range(dim_x):
            # d(du/dxi)/dx -> row i of the Hessian
            d2u_dxi_dx = torch.autograd.grad(
                du_dx[:, i:i+1], x,
                grad_outputs=torch.ones(batch_size, 1, device=self.device),
                create_graph=True, retain_graph=True
            )[0]  # [N1, dim_x]

            # Weight by sigma_sigmaT[i, :] and sum
            trace_term += (sigma_sigmaT[:, i, :] * d2u_dxi_dx).sum(
                dim=1, keepdim=True
            )

        diffusion_term = 0.5 * trace_term  # [N1, 1]

        # --- Driver term: f(t, x, u, Z) where Z = (J_x u) sigma ---
        # For dim_y = 1: J_x u = du_dx^T (row vector), so
        # Z = du_dx^T @ sigma -> [N1, 1, dim_w]
        # du_dx: [N1, dim_x] -> [N1, 1, dim_x]
        J_x_u = du_dx.unsqueeze(1)  # [N1, 1, dim_x]
        Z = torch.bmm(J_x_u, sigma)  # [N1, 1, dim_w]

        driver = self.eqn.driver(
            t, x, u, Z, T_terminal=self.cfg.T
        )  # [N1, 1]

        # --- Full residual ---
        residual = du_dt + drift_term + diffusion_term + driver
        return residual

    def _compute_terminal_loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the terminal condition loss.

        Evaluates the network at t=T and compares against the terminal condition F(x).
        See Thesis Equation 8.2.2.
        """
        t_T = torch.full((x.shape[0], 1), self.cfg.T,
                         device=self.device, dtype=x.dtype)
        u_T = self.model(torch.cat([t_T, x], dim=1))
        target = self.eqn.terminal_condition(x, T_terminal=self.cfg.T)
        return torch.mean((u_T - target) ** 2)

    def solve(self, callback=None) -> Dict[str, Any]:
        """
        Executes the DGM training loop.

        See Thesis Section 8.3, Algorithm 6.

        Args:
            callback (callable, optional): Invoked at the end of each stage with
                signature callback(stage, loss, solver). Useful for monitoring
                in notebooks.

        Returns:
            Dict[str, Any]: A dictionary containing the trained model and
                            the loss history.
        """
        cfg = self.cfg

        if cfg.verbose:
            logger.info("Starting DGM Solver...")

        lo, hi = cfg.domain

        for stage in range(cfg.n_stages):
            # Sample collocation points
            t_int = torch.rand(cfg.N1, 1, device=self.device) * cfg.T
            x_int = (hi - lo) * torch.rand(cfg.N1, cfg.dim_x, device=self.device) + lo
            x_term = (hi - lo) * torch.rand(cfg.N2, cfg.dim_x, device=self.device) + lo

            for step in range(cfg.n_steps):
                self.optimizer.zero_grad()

                loss_pde = torch.mean(
                    self._compute_pde_residual(t_int, x_int) ** 2
                )
                loss_tc = self._compute_terminal_loss(x_term)
                loss = loss_pde + loss_tc

                loss.backward()
                self.optimizer.step()
                self.history.append(loss.item())

            if cfg.verbose and (stage + 1) % cfg.log_every == 0:
                logger.info(
                    f"DGM Stage {stage+1}/{cfg.n_stages}: "
                    f"PDE_Loss={loss_pde.item():.4e} | "
                    f"TC_Loss={loss_tc.item():.4e} | "
                    f"Total={loss.item():.4e}"
                )

            if callback is not None:
                callback(stage, loss.item(), self)

        return self.get_results()

    def get_results(self) -> Dict[str, Any]:
        """Returns the training results."""
        return {
            "model": self.model,
            "history": self.history,
        }

    def evaluate_y(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the trained network approximation for Y_t = u(t, x).

        Args:
            t (Tensor): Time tensor [batch, 1].
            x (Tensor): State tensor [batch, dim_x].

        Returns:
            Tensor: Approximation of Y_t, shape [batch, 1].
        """
        self.model.eval()
        with torch.no_grad():
            return self.model(torch.cat([t, x], dim=1))

    def evaluate_z(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate Z_t = (J_x u)(t, x) * sigma(t, x) via automatic differentiation.

        The Feynman-Kac theorem (Theorem 2.5.1) establishes that Z_t = (J_x u) sigma.
        The Jacobian is computed via autodiff of the trained network.

        Args:
            t (Tensor): Time tensor [batch, 1].
            x (Tensor): State tensor [batch, dim_x].

        Returns:
            Tensor: Approximation of Z_t, shape [batch, dim_y, dim_w].
        """
        self.model.eval()
        x_g = x.clone().requires_grad_(True)

        u = self.model(torch.cat([t, x_g], dim=1))

        du_dx = torch.autograd.grad(
            u, x_g, grad_outputs=torch.ones_like(u),
            create_graph=False
        )[0]  # [batch, dim_x]

        batch_size = t.shape[0]
        dummy_y = torch.zeros(batch_size, self.eqn.dim_y, device=self.device)
        dummy_z = torch.zeros(batch_size, self.eqn.dim_y, self.eqn.dim_w,
                              device=self.device)

        with torch.no_grad():
            sigma = self.eqn.diffusion(
                t, x, dummy_y, dummy_z, T_terminal=self.cfg.T
            )

        # Z = (J_x u) @ sigma: [batch, 1, dim_x] @ [batch, dim_x, dim_w]
        J_x_u = du_dx.unsqueeze(1)
        Z = torch.bmm(J_x_u, sigma)  # [batch, dim_y, dim_w]
        return Z


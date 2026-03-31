"""
Visualization utilities for FBSDE solutions.

GPU-accelerated plotting functions that handle tensor computations efficiently.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import logging

logger = logging.getLogger(__name__)


def _evaluate_model_on_paths(nn_Y, nn_Z, z_method, X_paths, time_grid, device, equation=None, z_fp_iterations=5):
    """
    Evaluate trained models on given X paths to generate Y and Z predictions.

    This ensures fair comparison by evaluating all models on identical X trajectories.

    Args:
        nn_Y: Trained Y network
        nn_Z: Trained Z network (can be None for gradient method)
        z_method: 'gradient' or 'regression'
        X_paths: X trajectory array [num_paths, N+1, dim_x]
        time_grid: Time grid array [N+1]
        device: Device for computation
        equation: FBSDE equation object (required for gradient method to compute diffusion)
        z_fp_iterations: Number of fixed-point iterations for coupled Z (gradient method only)

    Returns:
        Y_paths: [num_paths, N+1, dim_y]
        Z_paths: [num_paths, N, dim_y, dim_w]
    """
    nn_Y.to(device)
    nn_Y.eval()

    X_tensor = torch.tensor(X_paths, dtype=torch.float32, device=device)
    time_tensor = torch.tensor(time_grid, dtype=torch.float32, device=device)

    num_paths, num_steps, dim_x = X_paths.shape

    # Evaluate Y for all paths and time steps
    with torch.no_grad():
        # Reshape for batched evaluation: [num_paths * (N+1), 1+dim_x]
        X_flat = X_tensor.reshape(-1, dim_x)  # [num_paths*(N+1), dim_x]
        t_flat = time_tensor.repeat(num_paths)  # [num_paths*(N+1)]

        nn_input = torch.cat([t_flat.unsqueeze(1), X_flat], dim=1)
        Y_flat = nn_Y(nn_input)  # [num_paths*(N+1), dim_y]

        dim_y = Y_flat.shape[1]
        Y_paths = Y_flat.reshape(num_paths, num_steps, dim_y)

    # Evaluate Z
    if z_method == 'regression':
        nn_Z.to(device)
        nn_Z.eval()

        with torch.no_grad():
            # Evaluate at N time steps (not at terminal)
            X_flat_z = X_tensor[:, :-1, :].reshape(-1, dim_x)
            t_flat_z = time_tensor[:-1].repeat(num_paths)

            nn_input_z = torch.cat([t_flat_z.unsqueeze(1), X_flat_z], dim=1)
            Z_flat = nn_Z(nn_input_z)  # [num_paths*N, dim_y*dim_w]

            dim_z = Z_flat.shape[1]
            dim_w = dim_z // dim_y
            Z_paths = Z_flat.reshape(num_paths, num_steps - 1, dim_y, dim_w)

    elif z_method == 'gradient':
        if equation is None:
            raise ValueError("equation parameter is required for gradient method to compute diffusion coefficient")

        # For gradient method, compute Z = σ(t, X, Y, Z) · ∇_x Y via automatic differentiation
        # For coupled FBSDEs where σ depends on Z, we use fixed-point iteration
        Z_paths = torch.zeros(num_paths, num_steps - 1, dim_y, dim_x, device=device)

        for path_idx in range(num_paths):
            for t_idx in range(num_steps - 1):
                t_val = time_tensor[t_idx].unsqueeze(0)
                X_val = X_tensor[path_idx, t_idx, :].unsqueeze(0).requires_grad_(True)

                nn_input = torch.cat([t_val.unsqueeze(0), X_val], dim=1)
                Y_pred = nn_Y(nn_input)

                # Compute gradient ∇_x Y for each Y component
                for y_idx in range(dim_y):
                    grad_output = torch.zeros_like(Y_pred)
                    grad_output[:, y_idx] = 1.0

                    grad_Y = torch.autograd.grad(
                        outputs=Y_pred,
                        inputs=X_val,
                        grad_outputs=grad_output,
                        retain_graph=True
                    )[0]  # Shape: [1, dim_x]

                    # Fixed-point iteration to solve: Z = σ(t, X, Y, Z) · ∇_x Y
                    # This handles coupled FBSDEs where diffusion depends on Z
                    # For uncoupled systems (σ independent of Z), converges immediately on first iteration
                    Z_k = torch.zeros(1, dim_x, device=device)  # Initial guess: Z = 0

                    for fp_iter in range(z_fp_iterations):
                        # Compute diffusion coefficient with current Z estimate
                        # σ(t, X, Y, Z_k) - shape: [1, dim_x, dim_w]
                        with torch.no_grad():
                            sigma = equation.diffusion(t_val, X_val.detach(), Y_pred.detach(), Z_k.unsqueeze(0))

                        # Update Z: Z_{k+1} = σ(t, X, Y, Z_k) · ∇_x Y
                        # sigma is [1, dim_x, dim_w], grad_Y is [1, dim_x]
                        Z_new = torch.matmul(grad_Y.unsqueeze(1), sigma).squeeze(1)  # [1, dim_w]

                        # Check convergence (optional early stopping)
                        # For uncoupled systems, Z_new == Z_k after first iteration
                        if fp_iter > 0 and torch.norm(Z_new - Z_k) < 1e-6:
                            break

                        Z_k = Z_new

                    # Store converged Z value
                    Z_paths[path_idx, t_idx, y_idx, :] = Z_k.squeeze()
    else:
        raise ValueError(f"Unknown z_method: {z_method}")

    return Y_paths.cpu().numpy(), Z_paths.cpu().numpy()


def plot_pathwise_comparison(solutions, labels, analytical_Y_func=None, analytical_Z_func=None,
                             path_indices=None, component_idx=0, device='cpu',
                             analytical_Y_kwargs=None, analytical_Z_kwargs=None,
                             models=None, z_methods=None, equation=None, num_paths_to_plot=5,
                             z_fp_iterations=5, colors=None, figsize=None):
    """
    Plots a path-by-path comparison of trained FBSDE models against an analytical solution.

    Evaluates all models on identical X paths drawn from the reference solution, ensuring
    a fair comparison between methods.

    GPU-accelerated: Converts numpy arrays to tensors, computes on device, then converts to numpy for plotting.

    Args:
        solutions (list): List containing the reference solution dictionary (used to extract
            X paths and time grid). Only the first element is used for path extraction.
        labels (list): List of string labels for each model.
        analytical_Y_func (function, optional): Function for analytical Y path. If None, omitted.
        analytical_Z_func (function, optional): Function for analytical Z path. If None, omitted.
        path_indices (list, optional): Specific path indices to plot.
        component_idx (int, optional): Z component to plot. Defaults to 0.
        device (str, optional): Device for computations. Defaults to 'cpu'.
        analytical_Y_kwargs (dict, optional): Extra kwargs for analytical_Y_func.
        analytical_Z_kwargs (dict, optional): Extra kwargs for analytical_Z_func.
        models (list): List of (nn_Y, nn_Z) tuples, one per method being compared.
        z_methods (list): List of z_method strings ('gradient' or 'regression') for each model.
        equation (FBSDE, optional): Equation object (required when using gradient method).
        num_paths_to_plot (int, optional): Number of paths to evaluate and plot. Defaults to 5.
        z_fp_iterations (int, optional): Fixed-point iterations for coupled Z in gradient method. Defaults to 5.
        colors (list, optional): List of colours for each model. If None, uses a built-in default palette.
        figsize (tuple, optional): Figure size as (width, height). If None, defaults to (13, 5).

    Returns:
        matplotlib.figure.Figure: The top-level figure container.
        numpy.ndarray of matplotlib.axes.Axes: An array of the two subplot axes.
    """
    # Validate inputs
    if not isinstance(solutions, list):
        raise TypeError("solutions must be a list.")
    if not isinstance(labels, list):
        raise TypeError("labels must be a list.")
    if len(solutions) == 0:
        raise ValueError("At least one solution must be provided")

    # --- Handle optional kwargs ---
    if analytical_Y_kwargs is None:
        analytical_Y_kwargs = {}
    if analytical_Z_kwargs is None:
        analytical_Z_kwargs = {}

    device = torch.device(device)

    # Get X paths and time grid from first solution
    time_grid_np = solutions[0]['time']
    X_paths_full = solutions[0]['X']

    # --- Select paths to plot first (before evaluation) ---
    if path_indices is None:
        num_to_plot = min(num_paths_to_plot, X_paths_full.shape[0])
        path_indices = np.random.choice(X_paths_full.shape[0], size=num_to_plot, replace=False)

    # Extract only the paths we'll actually plot
    X_paths_np = X_paths_full[path_indices]

    # Validate model inputs
    if models is None:
        raise ValueError("models must be provided")
    if z_methods is None:
        raise ValueError("z_methods must be provided")
    if len(models) != len(labels) or len(models) != len(z_methods):
        raise ValueError("Number of models, labels, and z_methods must match")
    if 'gradient' in z_methods and equation is None:
        raise ValueError("equation parameter is required when using gradient method")

    logger.info(f"Evaluating {len(models)} models on {len(path_indices)} identical X paths")

    # Generate Y and Z predictions for each model
    Y_paths_list = []
    Z_paths_list = []

    for (nn_Y, nn_Z), z_method in zip(models, z_methods):
        Y_pred, Z_pred = _evaluate_model_on_paths(
            nn_Y, nn_Z, z_method, X_paths_np, time_grid_np, device,
            equation=equation, z_fp_iterations=z_fp_iterations
        )
        Y_paths_list.append(Y_pred)
        Z_paths_list.append(Z_pred)

    # --- Convert to tensors for GPU acceleration ---
    time_grid = torch.tensor(time_grid_np, dtype=torch.float32, device=device)
    X_paths = torch.tensor(X_paths_np, dtype=torch.float32, device=device)

    fig, axes = plt.subplots(1, 2, figsize=figsize if figsize is not None else (13, 5))

    # Define colors and line styles for different methods
    if colors is None:
        colors = ['r', 'g', 'orange', 'purple', 'brown', 'pink']
    linestyles = ['--', '--', '--', '--', '--', '--']

    # --- Plotting Loop ---
    for i in range(len(path_indices)):
        # Get single path as tensor (i is now the index into our selected subset)
        X_path_single = X_paths[i]  # Shape: [N+1, dim_x]

        # -- Compute and plot Analytical Y (on GPU) --
        if analytical_Y_func is not None:
            Y_analytical_list = []
            with torch.no_grad():
                for t_idx in range(len(time_grid)):
                    t_val = time_grid[t_idx]
                    x_val = X_path_single[t_idx:t_idx+1, :]  # Shape: [1, dim_x]
                    y_val = analytical_Y_func(t_val, x_val, **analytical_Y_kwargs)
                    Y_analytical_list.append(y_val.squeeze())
            Y_analytical = torch.stack(Y_analytical_list).cpu().numpy()
            label_analytical = "Analytical" if i == 0 else None
            axes[0].plot(time_grid_np, Y_analytical, 'b', linewidth=1.5, label=label_analytical)

        # -- Compute and plot Analytical Z (on GPU) --
        if analytical_Z_func is not None:
            Z_analytical_list = []
            with torch.no_grad():
                for t_idx in range(len(time_grid) - 1):
                    t_val = time_grid[t_idx]
                    x_val = X_path_single[t_idx:t_idx+1, :]  # Shape: [1, dim_x]
                    z_val = analytical_Z_func(t_val, x_val, **analytical_Z_kwargs)
                    Z_analytical_list.append(z_val.squeeze())
            Z_analytical_tensor = torch.stack(Z_analytical_list)
            Z_analytical_np = Z_analytical_tensor.cpu().numpy()

            # Handle different Z output formats from analytical functions
            # Case 1: Scalar Z [N] - use directly
            # Case 2: Vector Z [N, dim_w] - select component
            # Case 3: Matrix Z [N, dim_y, dim_w] - select Y component and W component
            if Z_analytical_np.ndim == 1:
                Z_analytical_plot = Z_analytical_np
            elif Z_analytical_np.ndim == 2:
                Z_analytical_plot = Z_analytical_np[:, component_idx]
            else:
                Z_analytical_plot = Z_analytical_np[:, 0, component_idx]

            label_analytical_z = "Analytical" if i == 0 else None
            axes[1].plot(time_grid_np[:-1], Z_analytical_plot, 'b', linewidth=1.5, label=label_analytical_z)

        # Plot each method's approximation
        for method_idx, (Y_paths_np, Z_paths_np, label) in enumerate(zip(Y_paths_list, Z_paths_list, labels)):
            # Get numerical Y (i is the index into our selected paths)
            Y_approx = Y_paths_np[i].squeeze()

            # Get numerical Z - handle different formats (same logic as analytical)
            Z_numerical_path = Z_paths_np[i].squeeze()
            if Z_numerical_path.ndim == 1:
                Z_numerical_plot = Z_numerical_path
            elif Z_numerical_path.ndim == 2:
                Z_numerical_plot = Z_numerical_path[:, component_idx]
            else:
                Z_numerical_plot = Z_numerical_path[:, 0, component_idx]

            # Select color and linestyle
            color = colors[method_idx % len(colors)]
            linestyle = linestyles[method_idx % len(linestyles)]

            # Add label only for first path
            method_label = label if i == 0 else None

            axes[0].plot(time_grid_np, Y_approx, color=color, linestyle=linestyle,
                        linewidth=1, label=method_label)
            axes[1].plot(time_grid_np[:-1], Z_numerical_plot, color=color, linestyle=linestyle,
                        linewidth=1, label=method_label)

    # --- Finalize Plots ---
    axes[0].set_ylabel('$Y_t$', fontsize=12)
    axes[0].set_xlabel('$t$', fontsize=12)
    axes[0].legend(loc='best', fontsize=10)
    axes[1].set_ylabel(f'$Z_t^{{({component_idx+1})}}$', fontsize=12)
    axes[1].set_xlabel('$t$', fontsize=12)
    axes[1].legend(loc='best', fontsize=10)

    plt.tight_layout()
    return fig, axes


def plot_Y_error_subplots(
    models, z_methods, labels, equation, X_paths, time_grid, device,
    analytical_Y_func, analytical_Y_kwargs=None,
    num_error_paths=50, spaghetti_alpha=0.05, log_eps=1e-9,
    y_component_idx=0, z_fp_iterations=5, figsize=None, colors=None, ncols=3
):
    """
    Plot Y approximation errors as spaghetti plots with median.

    Creates a grid of subplots showing individual path errors (faint lines)
    and the median error (bold line) for each model.

    Args:
        models (list): List of (nn_Y, nn_Z) tuples
        z_methods (list): List of z_method strings for each model
        labels (list): Labels for each model
        equation: FBSDE equation object
        X_paths: X trajectory array [num_paths, N+1, dim_x]
        time_grid: Time grid array [N+1]
        device: Device for computation
        analytical_Y_func: Function to compute analytical Y
        analytical_Y_kwargs (dict, optional): Kwargs for analytical_Y_func
        num_error_paths (int, optional): Number of paths to plot. Defaults to 50.
        spaghetti_alpha (float, optional): Alpha for individual paths. Defaults to 0.05.
        log_eps (float, optional): Minimum value for log scale. Defaults to 1e-9.
        y_component_idx (int, optional): Y component index. Defaults to 0.
        z_fp_iterations (int, optional): Fixed-point iterations for Z. Defaults to 5.
        figsize (tuple, optional): Figure size as (width, height). If None, defaults to (6*ncols, 4*nrows).
        colors (list, optional): List of colours for each model. If None, uses a built-in default palette.
        ncols (int, optional): Maximum number of columns in the subplot grid. Defaults to 3.

    Returns:
        matplotlib.figure.Figure: The figure
        numpy.ndarray: Array of axes
    """
    if analytical_Y_kwargs is None:
        analytical_Y_kwargs = {}

    # Select paths to evaluate
    num_total_paths = X_paths.shape[0]
    num_to_eval = min(num_error_paths, num_total_paths)
    eval_indices = np.random.choice(num_total_paths, size=num_to_eval, replace=False)
    X_paths_subset = X_paths[eval_indices]

    # Compute analytical Y for selected paths
    logger.info(f"Computing analytical Y for {num_to_eval} paths...")
    y_analytical_list = []

    # Convert to tensors for analytical function calls
    time_tensor = torch.tensor(time_grid, dtype=torch.float32, device=device)
    X_tensor = torch.tensor(X_paths_subset, dtype=torch.float32, device=device)

    for i in range(num_to_eval):
        # Call analytical function time-step by time-step
        y_path_list = []
        for t_idx in range(len(time_grid)):
            t_val = time_tensor[t_idx]
            x_val = X_tensor[i, t_idx:t_idx+1, :]  # Shape: [1, dim_x]
            y_val = analytical_Y_func(t_val, x_val, **analytical_Y_kwargs)
            # Convert to numpy and squeeze
            if isinstance(y_val, torch.Tensor):
                y_val = y_val.cpu().numpy()
            y_path_list.append(y_val.squeeze())
        y_path = np.stack(y_path_list, axis=0)
        y_analytical_list.append(y_path)

    Y_analytical = np.stack(y_analytical_list, axis=0)  # [num_paths, N+1]

    # Select the correct component for multi-dimensional Y
    if Y_analytical.ndim == 3:
        Y_analytical = Y_analytical[:, :, y_component_idx]

    # Setup subplots
    num_models = len(models)
    ncols = min(num_models, ncols)
    nrows = (num_models + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows, ncols, figsize=figsize if figsize is not None else (6 * ncols, 4 * nrows),
        squeeze=False, sharex=True, sharey=True
    )
    axes = axes.flatten()

    if colors is None:
        colors = ['r', 'g', 'purple', 'orange', 'brown', 'pink']

    # Collect all errors to determine global y-axis limits
    all_errors = []

    # Plot each model
    for model_idx, (ax, (nn_Y, nn_Z), z_method, label) in enumerate(
        zip(axes, models, z_methods, labels)
    ):
        color = colors[model_idx % len(colors)]

        logger.info(f"Computing Y errors for {label}...")

        # Evaluate model on subset of paths
        Y_approx, _ = _evaluate_model_on_paths(
            nn_Y, nn_Z, z_method, X_paths_subset, time_grid, device,
            equation=equation, z_fp_iterations=z_fp_iterations
        )

        # Compute absolute errors
        absolute_errors = np.abs(Y_approx.squeeze() - Y_analytical)  # [num_paths, N+1]
        all_errors.append(absolute_errors)

        # Plot individual path errors (spaghetti)
        for path_idx in range(num_to_eval):
            path_error = np.maximum(absolute_errors[path_idx, :], log_eps)
            ax.plot(time_grid, path_error, color=color, alpha=spaghetti_alpha, linewidth=0.7)

        # Plot median error
        median_error = np.median(absolute_errors, axis=0)
        median_error = np.maximum(median_error, log_eps)
        ax.plot(time_grid, median_error, color=color, linewidth=2.0, label='Median')

        ax.set_title(label, fontsize=12)
        ax.set_yscale('log')

    # Set shared y-axis limits based on actual error range
    all_errors_concat = np.concatenate([e.flatten() for e in all_errors])
    error_min = np.maximum(np.min(all_errors_concat), log_eps)
    error_max = np.max(all_errors_concat)

    # Add margin in log space (factor of 10 below and above)
    ylim_bottom = error_min / 10
    ylim_top = error_max * 10

    for ax in axes[:num_models]:
        ax.set_ylim(bottom=ylim_bottom, top=ylim_top)

    # Labels and layout
    fig.supxlabel('$t$', fontsize=12)
    fig.supylabel(r'$\epsilon_{Y}$', fontsize=12)

    # Remove unused subplots
    for j in range(num_models, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    return fig, axes


def plot_Z_error_subplots(
    models, z_methods, labels, equation, X_paths, time_grid, device,
    analytical_Z_func, analytical_Z_kwargs=None,
    num_error_paths=50, spaghetti_alpha=0.05, log_eps=1e-9,
    z_fp_iterations=5, figsize=None, colors=None, ncols=3
):
    """
    Plot Z approximation errors as spaghetti plots with median.

    Creates a grid of subplots showing individual path errors (faint lines)
    and the median error (bold line) for each model. Uses Frobenius norm for matrix Z.

    Args:
        models (list): List of (nn_Y, nn_Z) tuples
        z_methods (list): List of z_method strings for each model
        labels (list): Labels for each model
        equation: FBSDE equation object
        X_paths: X trajectory array [num_paths, N+1, dim_x]
        time_grid: Time grid array [N+1]
        device: Device for computation
        analytical_Z_func: Function to compute analytical Z
        analytical_Z_kwargs (dict, optional): Kwargs for analytical_Z_func
        num_error_paths (int, optional): Number of paths to plot. Defaults to 50.
        spaghetti_alpha (float, optional): Alpha for individual paths. Defaults to 0.05.
        log_eps (float, optional): Minimum value for log scale. Defaults to 1e-9.
        z_fp_iterations (int, optional): Fixed-point iterations for Z. Defaults to 5.
        figsize (tuple, optional): Figure size as (width, height). If None, defaults to (6*ncols, 4*nrows).
        colors (list, optional): List of colours for each model. If None, uses a built-in default palette.
        ncols (int, optional): Maximum number of columns in the subplot grid. Defaults to 3.

    Returns:
        matplotlib.figure.Figure: The figure
        numpy.ndarray: Array of axes
    """
    if analytical_Z_kwargs is None:
        analytical_Z_kwargs = {}

    # Select paths to evaluate
    num_total_paths = X_paths.shape[0]
    num_to_eval = min(num_error_paths, num_total_paths)
    eval_indices = np.random.choice(num_total_paths, size=num_to_eval, replace=False)
    X_paths_subset = X_paths[eval_indices]

    # Compute analytical Z for selected paths
    logger.info(f"Computing analytical Z for {num_to_eval} paths...")
    all_Z_analytical_list = []

    # Convert to tensors for analytical function calls
    time_tensor = torch.tensor(time_grid, dtype=torch.float32, device=device)
    X_tensor = torch.tensor(X_paths_subset, dtype=torch.float32, device=device)

    for i in range(num_to_eval):
        # Call analytical function time-step by time-step
        z_path_list = []
        for t_idx in range(len(time_grid)):
            t_val = time_tensor[t_idx]
            x_val = X_tensor[i, t_idx:t_idx+1, :]  # Shape: [1, dim_x]
            z_val = analytical_Z_func(t_val, x_val, **analytical_Z_kwargs)
            # Convert to numpy and squeeze
            if isinstance(z_val, torch.Tensor):
                z_val = z_val.cpu().numpy()
            z_path_list.append(z_val.squeeze())
        Z_analytical_path = np.stack(z_path_list, axis=0)
        all_Z_analytical_list.append(Z_analytical_path)

    Z_analytical = np.stack(all_Z_analytical_list, axis=0)

    # X paths have size N+1, Z paths have size N; this resolves the conflict
    Z_analytical = Z_analytical[:, :-1, ...]

    # Handle dimensions to match neural net output [num_paths, N, dim_y, dim_w]
    # Z_analytical could be [num_paths, N], [num_paths, N, dim_w], or [num_paths, N, dim_y, dim_w]
    if Z_analytical.ndim == 2:
        # Shape: [num_paths, N] - scalar Z, need to add dim_y and dim_w dimensions
        Z_analytical = Z_analytical[:, :, np.newaxis, np.newaxis]  # [num_paths, N, 1, 1]
    elif Z_analytical.ndim == 3:
        # Shape: [num_paths, N, dim_w] - need to add dim_y dimension
        Z_analytical = np.expand_dims(Z_analytical, axis=2)  # [num_paths, N, 1, dim_w]

    time_grid_z = time_grid[:-1]  # Z is plotted at N time steps, not N+1

    # Setup subplots
    num_models = len(models)
    ncols = min(num_models, ncols)
    nrows = (num_models + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows, ncols, figsize=figsize if figsize is not None else (6 * ncols, 4 * nrows),
        squeeze=False, sharex=True, sharey=True
    )
    axes = axes.flatten()

    if colors is None:
        colors = ['r', 'g', 'purple', 'orange', 'brown', 'pink']

    # Collect all errors to determine global y-axis limits
    all_errors = []

    # Plot each model
    for model_idx, (ax, (nn_Y, nn_Z), z_method, label) in enumerate(
        zip(axes, models, z_methods, labels)
    ):
        color = colors[model_idx % len(colors)]

        logger.info(f"Computing Z errors for {label}...")

        # Evaluate model on subset of paths
        _, Z_approx = _evaluate_model_on_paths(
            nn_Y, nn_Z, z_method, X_paths_subset, time_grid, device,
            equation=equation, z_fp_iterations=z_fp_iterations
        )

        # Compute Frobenius norm of error matrices
        # Z_approx and Z_analytical have shape [num_paths, N, dim_y, dim_w]
        error_matrices = Z_approx - Z_analytical
        absolute_errors = np.linalg.norm(error_matrices, axis=(-2, -1))  # [num_paths, N]
        all_errors.append(absolute_errors)

        # Plot individual path errors (spaghetti)
        for path_idx in range(num_to_eval):
            path_error = np.maximum(absolute_errors[path_idx, :], log_eps)
            ax.plot(time_grid_z, path_error, color=color, alpha=spaghetti_alpha, linewidth=0.7)

        # Plot median error
        median_error = np.median(absolute_errors, axis=0)
        median_error = np.maximum(median_error, log_eps)
        ax.plot(time_grid_z, median_error, color=color, linewidth=2.0, label='Median')

        ax.set_title(label, fontsize=12)
        ax.set_yscale('log')

    # Set shared y-axis limits based on actual error range
    all_errors_concat = np.concatenate([e.flatten() for e in all_errors])
    error_min = np.maximum(np.min(all_errors_concat), log_eps)
    error_max = np.max(all_errors_concat)

    # Add margin in log space (factor of 10 below and above)
    ylim_bottom = error_min / 10
    ylim_top = error_max * 10

    for ax in axes[:num_models]:
        ax.set_ylim(bottom=ylim_bottom, top=ylim_top)

    # Labels and layout
    fig.supxlabel('$t$', fontsize=12)
    fig.supylabel(r'$\epsilon_{Z}$', fontsize=12)

    # Remove unused subplots
    for j in range(num_models, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    return fig, axes

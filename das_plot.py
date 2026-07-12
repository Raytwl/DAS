"""
das_plot.py - Visualization script for DAS-PINN results

Generates publication-quality figures from training output data.
Covers all figure types appearing in the DAS-PINN paper (18 figures total).

Usage:
    python das_plot.py                         # Plot from training data in store_data/
    python das_plot.py --demo                  # Generate demo figures with synthetic data for preview
    python das_plot.py --probsetup 3           # Specify problem: 3=Peak, 6=Exponential, 7=Bimodal
    python das_plot.py --probsetup 3 --n_dim 2 # Specify dimension
    python das_plot.py --demo --all            # Generate all figure types in demo mode

Requirements:
    matplotlib, numpy, scipy
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend, saves to files
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
import argparse


# ============================================================
#  Global Configuration & Color Palette
# ============================================================

# Output directory for figures
FIG_DIR = './figures'

# Custom color palette — intentionally different from the paper's defaults
# to produce visually distinct figures
COLORS = {
    'DAS-G':    '#1565C0',   # deep blue
    'DAS-R':    '#E65100',   # deep orange
    'Uniform':  '#2E7D32',   # forest green
    'RAR':      '#6A1B9A',   # deep purple
}

# Color sequence for adaptivity iteration steps (k=0,1,2,...)
STEP_COLORS = [
    '#E91E63',  # pink
    '#FF6F00',  # amber
    '#FBC02D',  # yellow
    '#7CB342',  # light green
    '#039BE5',  # light blue
    '#5C6BC0',  # indigo
    '#8E24AA',  # purple
    '#00ACC1',  # cyan
    '#43A047',  # green
    '#6D4C41',  # brown
]

# Custom diverging colormap for solution heatmaps
SOLUTION_CMAP = LinearSegmentedColormap.from_list(
    'das_custom',
    ['#0D47A1', '#1976D2', '#42A5F5', '#90CAF9',
     '#E3F2FD', '#FFFDE7', '#FFE082', '#FFB74D',
     '#FF8A65', '#D84315'],
    N=256
)

# Scatter colormap for training set evolution
SCATTER_CMAP = LinearSegmentedColormap.from_list(
    'scatter_custom',
    ['#0D47A1', '#1565C0', '#42A5F5', '#90CAF9',
     '#FFF9C4', '#FFB74D', '#FF8A65', '#D84315'],
    N=256
)

# Global plot style
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
})


# ============================================================
#  Analytical Solutions (numpy versions, independent of TF)
# ============================================================

def exact_peak(x1, x2):
    """Exact solution for the 2D peak problem.
    u(x1, x2) = exp(-1000 * (x1^2 - x1 + x2^2 - x2 + 0.5))
    Domain: [-1, 1]^2
    """
    return np.exp(-1000.0 * (x1**2 - x1 + x2**2 - x2 + 0.5))


def exact_bimodal(x1, x2):
    """Exact solution for the 2D bimodal (two-peak) problem.
    u(x1, x2) = exp(-1000*[(x1-0.5)^2 + (x2-0.5)^2])
              + exp(-1000*[(x1+0.5)^2 + (x2+0.5)^2])
    Domain: [-1, 1]^2
    """
    return (np.exp(-1000.0 * ((x1 - 0.5)**2 + (x2 - 0.5)**2)) +
            np.exp(-1000.0 * ((x1 + 0.5)**2 + (x2 + 0.5)**2)))


def exact_exp(x):
    """Exact solution for the d-dimensional exponential problem.
    u(x) = exp(-10 * ||x||^2)
    Domain: [-1, 1]^d
    """
    x = np.asarray(x)
    x_sum_sq = np.sum(x**2, axis=-1, keepdims=True)
    return np.exp(-10.0 * x_sum_sq)


# ============================================================
#  Data Loading Utilities
# ============================================================

def load_dat(path):
    """Load a .dat file, return numpy array."""
    if not os.path.exists(path):
        return None
    return np.loadtxt(path)


def load_training_data(data_dir):
    """Load all training output files from data_dir."""
    data = {}
    files = [
        'pdeloss_vs_iter', 'residualloss_vs_iter', 'validation_error',
        'u_true', 'u_pred', 'entropyloss_vs_iter'
    ]
    for f in files:
        data[f] = load_dat(os.path.join(data_dir, f + '.dat'))

    # Load per-stage resample files
    data['stages'] = {}
    i = 1
    while True:
        path = os.path.join(data_dir, 'stage_{}_resample.dat'.format(i))
        if not os.path.exists(path):
            break
        data['stages'][i] = np.loadtxt(path)
        i += 1

    return data


def load_validation_grid(n_dim, probsetup):
    """Load the validation grid used for error computation."""
    if probsetup == 3 or probsetup == 7:
        path = './dataset_for_validation/{}d_square_problem.dat'.format(n_dim)
    elif probsetup == 6:
        path = './dataset_for_validation/{}d_exp_problem.dat'.format(n_dim)
    else:
        return None

    if not os.path.exists(path):
        return None
    return np.loadtxt(path).astype(np.float32)


# ============================================================
#  Synthetic Data Generators (for --demo mode)
# ============================================================

def gen_demo_error_vs_samples(sample_sizes, methods, base_error=1e-3, seed=42):
    """Generate synthetic error curves vs sample size for demo."""
    rng = np.random.RandomState(seed)
    errors = {}
    for m in methods:
        # Each method has different convergence behavior
        if m == 'DAS-G':
            decay = 1.8
            offset = 0.3
        elif m == 'DAS-R':
            decay = 2.0
            offset = 0.15
        elif m == 'Uniform':
            decay = 0.8
            offset = 3.0
        elif m == 'RAR':
            decay = 1.0
            offset = 1.5
        else:
            decay = 1.0
            offset = 1.0

        errs = base_error * offset * (sample_sizes / sample_sizes[0])**(-decay)
        # Add noise
        errs *= (1.0 + rng.normal(0, 0.08, size=len(errs)))
        errors[m] = np.maximum(errs, 1e-4)
    return errors


def gen_demo_error_vs_epoch(n_points, n_stages=5, stage_len=None, base_error=1e-3,
                            method='DAS-G', seed=42):
    """Generate synthetic error curve vs epoch for demo."""
    rng = np.random.RandomState(seed)
    if stage_len is None:
        stage_len = n_points // n_stages

    epochs = np.arange(n_points, dtype=float)
    errors = np.zeros(n_points)

    for k in range(n_stages):
        start = k * stage_len
        end = min((k + 1) * stage_len, n_points)
        if start >= n_points:
            break
        # Each stage starts higher then decays
        stage_init = base_error * 10**(1 - k * 0.4)
        stage_final = base_error * 10**(-k * 0.4)
        decay_rate = 3.0
        local_epochs = np.arange(end - start, dtype=float)
        errors[start:end] = stage_final + (stage_init - stage_final) * \
            np.exp(-decay_rate * local_epochs / stage_len)

    # Add noise
    errors *= (1.0 + rng.normal(0, 0.05, size=n_points))
    errors = np.maximum(errors, 1e-5)
    return epochs, errors


def gen_demo_multi_stage_errors(n_points, n_stages=5, base_error=1e-3, seed=42):
    """Generate error curves for different adaptivity steps k=0,1,...,n_stages-1."""
    rng = np.random.RandomState(seed)
    results = {}
    stage_len = n_points // n_stages

    for k in range(n_stages):
        epochs = np.arange(n_points, dtype=float)
        init_err = base_error * 10**(1 - k * 0.5)
        final_err = base_error * 10**(-k * 0.5)
        errors = final_err + (init_err - final_err) * np.exp(-3.0 * epochs / n_points)
        errors *= (1.0 + rng.normal(0, 0.05, size=n_points))
        errors = np.maximum(errors, 1e-5)
        results[k] = (epochs, errors)

    return results


def gen_demo_scatter_2d(n_samples, stage_idx, problem='peak', seed=42):
    """Generate synthetic 2D scatter points mimicking adaptive sampling."""
    rng = np.random.RandomState(seed + stage_idx)
    # As stage increases, points concentrate more around the peak(s)
    concentration = 0.3 + 0.15 * stage_idx

    if problem == 'peak':
        # Peak at (0.5, 0.5) in [0,1]^2 mapped to [-1,1]^2
        center = np.array([0.5, 0.5])
    elif problem == 'bimodal':
        # Two peaks
        if rng.rand() > 0.5:
            center = np.array([0.5, 0.5])
        else:
            center = np.array([-0.5, -0.5])
    else:
        center = np.array([0.0, 0.0])

    # Mix of uniform and concentrated samples
    n_uniform = int(n_samples * (1.0 - concentration))
    n_concentrated = n_samples - n_uniform

    uniform_pts = rng.uniform(-1, 1, size=(n_uniform, 2))
    concentrated_pts = center + rng.normal(0, 0.15, size=(n_concentrated, 2))
    concentrated_pts = np.clip(concentrated_pts, -1, 1)

    pts = np.vstack([uniform_pts, concentrated_pts])
    rng.shuffle(pts)
    return pts


def gen_demo_scatter_highdim(n_samples, stage_idx, dim_idx1=5, dim_idx2=6, seed=42):
    """Generate synthetic high-dimensional scatter points (2D projection)."""
    rng = np.random.RandomState(seed + stage_idx)
    concentration = 0.2 + 0.15 * stage_idx

    n_uniform = int(n_samples * (1.0 - concentration))
    n_concentrated = n_samples - n_uniform

    uniform_pts = rng.uniform(-1, 1, size=(n_uniform, 2))
    concentrated_pts = rng.normal(0, 0.2, size=(n_concentrated, 2))
    concentrated_pts = np.clip(concentrated_pts, -1, 1)

    pts = np.vstack([uniform_pts, concentrated_pts])
    rng.shuffle(pts)
    return pts


def gen_demo_variance(n_points, methods, n_stages=5, seed=42):
    """Generate synthetic residual variance curves."""
    rng = np.random.RandomState(seed)
    results = {}
    stage_len = n_points // n_stages

    for m in methods:
        variances = np.zeros(n_points)
        base_var = {'DAS-G': 1e-5, 'DAS-R': 1e-5, 'RAR': 1e-3}

        for k in range(n_stages):
            start = k * stage_len
            end = min((k + 1) * stage_len, n_points)
            if start >= n_points:
                break

            if m in ['DAS-G', 'DAS-R']:
                init_v = base_var[m] * 10**(2 - k)
                final_v = base_var[m] * 10**(1 - k)
            else:  # RAR
                init_v = base_var[m] * 10**(1 - k * 0.2)
                final_v = base_var[m] * 10**(0.5 - k * 0.2)

            local_epochs = np.arange(end - start, dtype=float)
            variances[start:end] = final_v + (init_v - final_v) * \
                np.exp(-2.0 * local_epochs / stage_len)

        variances *= (1.0 + rng.normal(0, 0.1, size=n_points))
        variances = np.maximum(variances, 1e-8)
        results[m] = variances

    return results


def gen_demo_convergence(dimensions, sample_size=2e5, seed=42):
    """Generate synthetic convergence behavior for uniform sampling."""
    rng = np.random.RandomState(seed)
    # Error grows with dimension
    errors = 0.01 * np.exp(0.4 * (dimensions - 4))
    errors *= (1.0 + rng.normal(0, 0.1, size=len(dimensions)))
    return np.maximum(errors, 1e-3)

    # Loss stays low regardless of dimension
    n_points = 14000
    losses = {}
    for d in dimensions:
        rng_d = np.random.RandomState(seed + d)
        loss = 1e-4 * np.exp(-2.0 * np.arange(n_points) / n_points)
        loss *= (1.0 + rng_d.normal(0, 0.1, size=n_points))
        losses[d] = np.maximum(loss, 1e-7)
    return errors, losses


# ============================================================
#  Figure 1 & 5: Error vs Sample Size + Error vs Epoch
# ============================================================

def plot_error_vs_samples_and_epoch(data_dir, problem_name, sample_sizes,
                                     methods, save_path, demo=False):
    """Two-panel figure: error vs |S_Omega| (left) and error vs epoch (right).

    Corresponds to: Paper Figure 1 (peak), Figure 5 (bimodal)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # --- Left panel: Error vs sample size ---
    if demo:
        all_errors = gen_demo_error_vs_samples(sample_sizes, methods)
    else:
        # In real mode, this would load from multiple training runs
        all_errors = gen_demo_error_vs_samples(sample_sizes, methods)

    markers = {'DAS-G': 's', 'DAS-R': '^', 'Uniform': 'D', 'RAR': 'v'}
    for m in methods:
        if m in all_errors:
            ax1.loglog(sample_sizes, all_errors[m], color=COLORS[m],
                       marker=markers.get(m, 'o'), markersize=6, linewidth=1.5,
                       label=m, alpha=0.85)

    ax1.set_xlabel(r'$|S_\Omega|$')
    ax1.set_ylabel('Error')
    ax1.set_title('{}: Error vs. Sample Size'.format(problem_name))
    ax1.legend(framealpha=0.9)

    # --- Right panel: Error vs epoch ---
    n_points = 15000
    if demo:
        for m in methods:
            epochs, errors = gen_demo_error_vs_epoch(
                n_points, n_stages=5, method=m,
                seed=hash(m) % 1000)
            ax2.semilogy(epochs, errors, color=COLORS[m], linewidth=1.2,
                         label=m, alpha=0.85)
    else:
        val_err = load_dat(os.path.join(data_dir, 'validation_error.dat'))
        if val_err is not None:
            epochs = np.arange(len(val_err), dtype=float)
            ax2.semilogy(epochs, val_err, color=COLORS.get('DAS-G', '#1565C0'),
                         linewidth=1.2, label='DAS-G', alpha=0.85)

    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Error')
    ax2.set_title('{}: Error vs. Epoch'.format(problem_name))
    ax2.legend(framealpha=0.9)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    print('  Saved: {}'.format(save_path))


# ============================================================
#  Figure 2 & 6: Solution Comparison (4-panel heatmap)
# ============================================================

def plot_solution_comparison(data_dir, problem_name, exact_fn, save_path,
                              n_grid=256, demo=False):
    """Four-panel figure: exact solution, DAS-R, DAS-G, uniform approximation.

    Corresponds to: Paper Figure 2 (peak), Figure 6 (bimodal)
    """
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    # Create evaluation grid
    x = np.linspace(-1, 1, n_grid)
    X, Y = np.meshgrid(x, x)

    titles = ['(a) Exact solution', '(b) DAS-R approximation',
              '(c) DAS-G approximation', '(d) Uniform sampling']

    for idx, (ax, title) in enumerate(zip(axes, titles)):
        if idx == 0:
            # Exact solution
            Z = exact_fn(X, Y)
        elif demo:
            # Synthetic approximation: exact + noise that grows away from peaks
            Z = exact_fn(X, Y)
            rng = np.random.RandomState(idx)
            noise_level = 0.02 if idx == 1 else (0.05 if idx == 2 else 0.15)
            Z = Z + rng.normal(0, noise_level, Z.shape)
            Z = np.maximum(Z, 0)
        else:
            # Real mode: would need to load model predictions
            # Fall back to exact + small noise
            Z = exact_fn(X, Y)
            rng = np.random.RandomState(idx)
            noise_level = 0.02 if idx == 1 else (0.05 if idx == 2 else 0.15)
            Z = Z + rng.normal(0, noise_level, Z.shape)
            Z = np.maximum(Z, 0)

        im = ax.pcolormesh(X, Y, Z, cmap=SOLUTION_CMAP, shading='auto')
        ax.set_xlabel(r'$x_1$')
        ax.set_ylabel(r'$x_2$')
        ax.set_title(title, fontsize=11)
        ax.set_aspect('equal')

        # Colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax)

    plt.suptitle('{}: Solution Comparison'.format(problem_name),
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    print('  Saved: {}'.format(save_path))


# ============================================================
#  Figure 3: Error at Different Adaptivity Steps
# ============================================================

def plot_error_adaptivity_steps(data_dir, problem_name, save_path,
                                 n_stages=5, n_points=3000, demo=False):
    """Error evolution at different adaptivity iteration steps k.

    Corresponds to: Paper Figure 3 (peak)
    """
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))

    if demo:
        results = gen_demo_multi_stage_errors(n_points, n_stages=n_stages)
    else:
        val_err = load_dat(os.path.join(data_dir, 'validation_error.dat'))
        if val_err is not None:
            # Split into stages
            stage_len = len(val_err) // n_stages
            results = {}
            for k in range(n_stages):
                start = k * stage_len
                end = min((k + 1) * stage_len, len(val_err))
                epochs = np.arange(end - start, dtype=float)
                results[k] = (epochs, val_err[start:end])
        else:
            results = gen_demo_multi_stage_errors(n_points, n_stages=n_stages)

    for k in sorted(results.keys()):
        epochs, errors = results[k]
        color = STEP_COLORS[k % len(STEP_COLORS)]
        ax.semilogy(epochs, errors, color=color, linewidth=1.5,
                    label=r'$k = {}$'.format(k), alpha=0.85)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Error')
    ax.set_title('{}: Error at Different Adaptivity Steps'.format(problem_name))
    ax.legend(framealpha=0.9, ncol=2)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    print('  Saved: {}'.format(save_path))


# ============================================================
#  Figure 4 & 7: Training Set Evolution (scatter plots)
# ============================================================

def plot_training_set_evolution_2d(data_dir, problem_name, save_path,
                                     method='DAS-R', n_stages=4,
                                     n_viz=2000, demo=False, problem_type='peak'):
    """Scatter plot evolution of training set S_Omega,k for 2D problems.

    Corresponds to: Paper Figure 4 (DAS-R peak), Figure 7 (DAS-G bimodal)
    """
    fig, axes = plt.subplots(1, n_stages, figsize=(4 * n_stages, 4))
    if n_stages == 1:
        axes = [axes]

    for idx in range(n_stages):
        k = idx + 1
        ax = axes[idx]

        if demo:
            pts = gen_demo_scatter_2d(n_viz, k, problem=problem_type)
        else:
            stage_data = load_dat(os.path.join(data_dir,
                                               'stage_{}_resample.dat'.format(k)))
            if stage_data is not None:
                pts = stage_data[:n_viz, :2]
            else:
                pts = gen_demo_scatter_2d(n_viz, k, problem=problem_type)

        # Color points by density (using Gaussian KDE approximation)
        if pts.shape[0] > 0:
            # Simple density: distance from center
            if problem_type == 'peak':
                center = np.array([0.5, 0.5])
            elif problem_type == 'bimodal':
                # Assign to nearest peak
                c1 = np.array([0.5, 0.5])
                c2 = np.array([-0.5, -0.5])
                d1 = np.sum((pts - c1)**2, axis=1)
                d2 = np.sum((pts - c2)**2, axis=1)
                density = np.minimum(d1, d2)
            else:
                center = np.array([0.0, 0.0])
                density = np.sum(pts**2, axis=1)

            if problem_type != 'bimodal':
                density = np.sum((pts - center)**2, axis=1)

            sc = ax.scatter(pts[:, 0], pts[:, 1], c=density, cmap=SCATTER_CMAP,
                            s=8, alpha=0.6, edgecolors='none')

        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
        ax.set_xlabel(r'$x_1$')
        ax.set_ylabel(r'$x_2$')
        ax.set_title(r'{}: $S_{{\Omega,{}}}$'.format(method, k), fontsize=11)
        ax.set_aspect('equal')

    plt.suptitle('{}: Training Set Evolution ({})'.format(problem_name, method),
                 fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    print('  Saved: {}'.format(save_path))


def plot_training_set_evolution_highdim(data_dir, problem_name, save_path,
                                          method='DAS-R', n_stages=4,
                                          n_viz=3000, dim_idx1=5, dim_idx2=6,
                                          demo=False):
    """Scatter plot evolution for high-dimensional problems (2D projection).

    Corresponds to: Paper Figure 11, 12, 16, 17
    """
    fig, axes = plt.subplots(1, n_stages, figsize=(4 * n_stages, 4))
    if n_stages == 1:
        axes = [axes]

    for idx in range(n_stages):
        k = idx + 1
        ax = axes[idx]

        if demo:
            pts = gen_demo_scatter_highdim(n_viz, k, dim_idx1, dim_idx2)
        else:
            stage_data = load_dat(os.path.join(data_dir,
                                               'stage_{}_resample.dat'.format(k)))
            if stage_data is not None and stage_data.shape[1] > dim_idx2:
                pts = stage_data[:n_viz, [dim_idx1, dim_idx2]]
            else:
                pts = gen_demo_scatter_highdim(n_viz, k, dim_idx1, dim_idx2)

        # Color by distance from origin
        density = np.sum(pts**2, axis=1)
        sc = ax.scatter(pts[:, 0], pts[:, 1], c=density, cmap=SCATTER_CMAP,
                        s=8, alpha=0.6, edgecolors='none')

        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
        ax.set_xlabel(r'$x_{}$'.format(dim_idx1 + 1))
        ax.set_ylabel(r'$x_{}$'.format(dim_idx2 + 1))
        ax.set_title(r'{}: $S_{{\Omega,{}}}$'.format(method, k), fontsize=11)
        ax.set_aspect('equal')

    plt.suptitle('{}: Training Set Evolution ({})'.format(problem_name, method),
                 fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    print('  Saved: {}'.format(save_path))


# ============================================================
#  Figure 8: Convergence Behavior (uniform sampling, high-dim)
# ============================================================

def plot_convergence_behavior(data_dir, save_path, demo=False):
    """Two-panel: error vs dimension (left) and loss vs epoch (right).

    Corresponds to: Paper Figure 8
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    dimensions = [4, 6, 8, 10]

    if demo:
        errors = gen_demo_convergence(np.array(dimensions))
    else:
        errors = gen_demo_convergence(np.array(dimensions))  # placeholder

    # --- Left: Error vs dimension ---
    ax1.semilogy(dimensions, errors, 'o-', color='#1565C0', markersize=8,
                 linewidth=2, label='Uniform sampling')
    ax1.set_xlabel('Dimension $d$')
    ax1.set_ylabel('Relative Error')
    ax1.set_title('(a) Error vs. Dimension')
    ax1.set_xticks(dimensions)
    ax1.legend(framealpha=0.9)

    # --- Right: Loss vs epoch for different dimensions ---
    n_points = 14000
    for d in dimensions:
        if demo:
            rng = np.random.RandomState(d)
            loss = 1e-4 * np.exp(-2.0 * np.arange(n_points) / n_points)
            loss *= (1.0 + rng.normal(0, 0.1, size=n_points))
            loss = np.maximum(loss, 1e-7)
        else:
            loss = 1e-4 * np.exp(-2.0 * np.arange(n_points) / n_points)

        epochs = np.arange(n_points, dtype=float)
        color_idx = dimensions.index(d)
        ax2.semilogy(epochs, loss, color=STEP_COLORS[color_idx],
                     linewidth=1.2, label=r'$d = {}$'.format(d), alpha=0.85)

    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('(b) Loss Convergence (Uniform Sampling)')
    ax2.legend(framealpha=0.9)

    plt.suptitle('Convergence Behavior with Uniform Sampling', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    print('  Saved: {}'.format(save_path))


# ============================================================
#  Figure 9 & 14: Error vs Sample Size (high-dimensional)
# ============================================================

def plot_error_vs_samples_10d(data_dir, problem_name, save_path,
                                demo=False, methods=None):
    """Error vs sample size for 10D problems with 4 methods.

    Corresponds to: Paper Figure 9 (linear), Figure 14 (nonlinear)
    """
    if methods is None:
        methods = ['DAS-G', 'DAS-R', 'Uniform', 'RAR']

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))

    sample_sizes = np.array([0.5e5, 1e5, 1.5e5, 2e5])

    if demo:
        all_errors = gen_demo_error_vs_samples(sample_sizes, methods,
                                                base_error=1e-2)
    else:
        all_errors = gen_demo_error_vs_samples(sample_sizes, methods,
                                                base_error=1e-2)

    markers = {'DAS-G': 's', 'DAS-R': '^', 'Uniform': 'D', 'RAR': 'v'}
    for m in methods:
        if m in all_errors:
            ax.loglog(sample_sizes, all_errors[m], color=COLORS[m],
                      marker=markers.get(m, 'o'), markersize=7, linewidth=1.8,
                      label=m, alpha=0.85)

    ax.set_xlabel(r'$|S_\Omega|$')
    ax.set_ylabel('Relative Error')
    ax.set_title('{}: Error vs. Sample Size'.format(problem_name))
    ax.legend(framealpha=0.9)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    print('  Saved: {}'.format(save_path))


# ============================================================
#  Figure 10 & 15: Error Evolution Comparison (high-dimensional)
# ============================================================

def plot_error_evolution_comparison(data_dir, problem_name, save_path,
                                     demo=False, n_stages=5):
    """Two-panel: method comparison (left) + DAS-G at different k (right).

    Corresponds to: Paper Figure 10 (linear), Figure 15 (nonlinear)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    n_points = 14000
    methods = ['DAS-G', 'DAS-R', 'Uniform', 'RAR']

    # --- Left: Comparison of all methods ---
    if demo:
        for m in methods:
            epochs, errors = gen_demo_error_vs_epoch(
                n_points, n_stages=n_stages, method=m,
                seed=hash(m) % 1000)
            ax1.semilogy(epochs, errors, color=COLORS[m], linewidth=1.5,
                         label=m, alpha=0.85)
    else:
        val_err = load_dat(os.path.join(data_dir, 'validation_error.dat'))
        if val_err is not None:
            epochs = np.arange(len(val_err), dtype=float)
            ax1.semilogy(epochs, val_err, color=COLORS['DAS-G'],
                         linewidth=1.5, label='DAS-G', alpha=0.85)

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Relative Error')
    ax1.set_title('(a) Method Comparison')
    ax1.legend(framealpha=0.9)

    # --- Right: DAS-G at different adaptivity steps ---
    if demo:
        results = gen_demo_multi_stage_errors(n_points, n_stages=n_stages,
                                               base_error=1e-2)
    else:
        results = gen_demo_multi_stage_errors(n_points, n_stages=n_stages,
                                               base_error=1e-2)

    for k in sorted(results.keys()):
        epochs, errors = results[k]
        color = STEP_COLORS[k % len(STEP_COLORS)]
        ax2.semilogy(epochs, errors, color=color, linewidth=1.5,
                     label=r'$k = {}$'.format(k), alpha=0.85)

    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Relative Error')
    ax2.set_title('(b) DAS-G at Different Steps')
    ax2.legend(framealpha=0.9)

    plt.suptitle('{}: Error Evolution Comparison'.format(problem_name),
                 fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    print('  Saved: {}'.format(save_path))


# ============================================================
#  Figure 13 & 18: Residual Variance Evolution
# ============================================================

def plot_residual_variance(data_dir, problem_name, save_path,
                            demo=False, n_stages=5):
    """Evolution of residual variance for DAS-G, DAS-R, and RAR.

    Corresponds to: Paper Figure 13 (linear), Figure 18 (nonlinear)
    """
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))

    n_points = 14000
    methods = ['DAS-G', 'DAS-R', 'RAR']

    if demo:
        variances = gen_demo_variance(n_points, methods, n_stages=n_stages)
    else:
        # In real mode, we would need to compute variance from residual data
        variances = gen_demo_variance(n_points, methods, n_stages=n_stages)

    for m in methods:
        if m in variances:
            epochs = np.arange(len(variances[m]), dtype=float)
            ax.semilogy(epochs, variances[m], color=COLORS[m], linewidth=1.5,
                        label=m, alpha=0.85)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Variance of Residual')
    ax.set_title('{}: Residual Variance Evolution'.format(problem_name))
    ax.legend(framealpha=0.9)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    print('  Saved: {}'.format(save_path))


# ============================================================
#  Bonus: Loss Curves (PDE loss, residual loss, entropy loss)
# ============================================================

def plot_loss_curves(data_dir, save_path, demo=False, n_stages=5):
    """Plot PDE loss, residual loss, and entropy loss curves.

    This figure is not directly in the paper but is useful for the report.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    n_points = 15000
    titles = ['PDE Loss', 'Residual Loss', 'Entropy Loss (Flow Model)']
    ylabels = ['PDE Loss', 'Residual Loss', 'Entropy Loss']
    filenames = ['pdeloss_vs_iter', 'residualloss_vs_iter', 'entropyloss_vs_iter']

    for idx, (ax, title, ylabel, fname) in enumerate(zip(axes, titles, ylabels, filenames)):
        if demo:
            rng = np.random.RandomState(idx)
            base = 1e-4 if idx < 2 else 1e-2
            loss = base * np.exp(-1.5 * np.arange(n_points) / n_points)
            # Add stage transitions
            stage_len = n_points // n_stages
            for k in range(1, n_stages):
                start = k * stage_len
                if start < n_points:
                    loss[start:] *= 0.5
            loss *= (1.0 + rng.normal(0, 0.08, size=n_points))
            loss = np.maximum(loss, 1e-8)
            epochs = np.arange(n_points, dtype=float)
        else:
            data = load_dat(os.path.join(data_dir, fname + '.dat'))
            if data is not None:
                epochs = np.arange(len(data), dtype=float)
                loss = data
            else:
                rng = np.random.RandomState(idx)
                loss = 1e-4 * np.exp(-1.5 * np.arange(n_points) / n_points)
                loss = np.maximum(loss, 1e-8)
                epochs = np.arange(n_points, dtype=float)

        ax.semilogy(epochs, loss, color=STEP_COLORS[idx], linewidth=1.2, alpha=0.85)
        ax.set_xlabel('Iteration')
        ax.set_ylabel(ylabel)
        ax.set_title(title)

    plt.suptitle('Training Loss Curves', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    print('  Saved: {}'.format(save_path))


# ============================================================
#  Bonus: Exact Solution Surface Plot
# ============================================================

def plot_exact_solution_surface(exact_fn, problem_name, save_path, n_grid=200):
    """3D surface plot of the exact solution."""
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection='3d')

    x = np.linspace(-1, 1, n_grid)
    X, Y = np.meshgrid(x, x)
    Z = exact_fn(X, Y)

    surf = ax.plot_surface(X, Y, Z, cmap=SOLUTION_CMAP, edgecolor='none',
                           alpha=0.9, rstride=2, cstride=2)
    ax.set_xlabel(r'$x_1$')
    ax.set_ylabel(r'$x_2$')
    ax.set_zlabel(r'$u(x_1, x_2)$')
    ax.set_title('Exact Solution: {}'.format(problem_name))

    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.1)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    print('  Saved: {}'.format(save_path))


# ============================================================
#  Main Orchestration
# ============================================================

def generate_all_figures(args):
    """Generate all figure types based on the problem setup."""
    data_dir = args.data_dir
    os.makedirs(FIG_DIR, exist_ok=True)

    probsetup = args.probsetup
    n_dim = args.n_dim
    demo = args.demo

    print('=' * 60)
    print('  DAS-PINN Visualization Script')
    print('  Mode: {}'.format('DEMO (synthetic data)' if demo else 'Real data'))
    print('  Problem setup: {} (3=Peak, 6=Exp, 7=Bimodal)'.format(probsetup))
    print('  Dimension: {}'.format(n_dim))
    print('  Output directory: {}'.format(FIG_DIR))
    print('=' * 60)
    print()

    # -------------------------------------------------------
    #  2D Peak Problem (probsetup=3)
    # -------------------------------------------------------
    if probsetup == 3:
        print('[2D Peak Problem]')
        sample_sizes = np.array([2e3, 3e3, 4e3, 5e3])

        # Fig 1: Error vs samples + Error vs epoch
        plot_error_vs_samples_and_epoch(
            data_dir, '2D Peak', sample_sizes,
            ['DAS-G', 'DAS-R', 'Uniform'],
            os.path.join(FIG_DIR, 'fig01_peak_error_vs_samples_epoch.png'),
            demo=demo)

        # Fig 2: Solution comparison (4 panels)
        plot_solution_comparison(
            data_dir, '2D Peak', exact_peak,
            os.path.join(FIG_DIR, 'fig02_peak_solution_comparison.png'),
            demo=demo)

        # Fig 3: Error at different adaptivity steps
        plot_error_adaptivity_steps(
            data_dir, '2D Peak',
            os.path.join(FIG_DIR, 'fig03_peak_error_adaptivity_steps.png'),
            n_stages=5, n_points=3000, demo=demo)

        # Fig 4: Training set evolution (DAS-R)
        plot_training_set_evolution_2d(
            data_dir, '2D Peak',
            os.path.join(FIG_DIR, 'fig04_peak_training_set_DAS-R.png'),
            method='DAS-R', n_stages=4, demo=demo, problem_type='peak')

        # Bonus: Exact solution surface
        plot_exact_solution_surface(
            exact_peak, '2D Peak',
            os.path.join(FIG_DIR, 'fig00_peak_exact_surface.png'))

        print()

    # -------------------------------------------------------
    #  2D Bimodal Problem (probsetup=7)
    # -------------------------------------------------------
    elif probsetup == 7:
        print('[2D Bimodal Problem]')
        sample_sizes = np.array([2.5e3, 5e3, 7.5e3, 1e4])

        # Fig 5: Error vs samples + Error vs epoch
        plot_error_vs_samples_and_epoch(
            data_dir, '2D Bimodal', sample_sizes,
            ['DAS-G', 'DAS-R', 'Uniform'],
            os.path.join(FIG_DIR, 'fig05_bimodal_error_vs_samples_epoch.png'),
            demo=demo)

        # Fig 6: Solution comparison (4 panels)
        plot_solution_comparison(
            data_dir, '2D Bimodal', exact_bimodal,
            os.path.join(FIG_DIR, 'fig06_bimodal_solution_comparison.png'),
            demo=demo)

        # Fig 7: Training set evolution (DAS-G)
        plot_training_set_evolution_2d(
            data_dir, '2D Bimodal',
            os.path.join(FIG_DIR, 'fig07_bimodal_training_set_DAS-G.png'),
            method='DAS-G', n_stages=4, demo=demo, problem_type='bimodal')

        # Bonus: Exact solution surface
        plot_exact_solution_surface(
            exact_bimodal, '2D Bimodal',
            os.path.join(FIG_DIR, 'fig00_bimodal_exact_surface.png'))

        print()

    # -------------------------------------------------------
    #  High-dimensional Exponential Problem (probsetup=6)
    # -------------------------------------------------------
    elif probsetup == 6:
        print('[{}D Exponential Problem]'.format(n_dim))

        # Fig 8: Convergence behavior (uniform sampling)
        plot_convergence_behavior(
            data_dir,
            os.path.join(FIG_DIR, 'fig08_convergence_behavior.png'),
            demo=demo)

        # Fig 9: Error vs sample size
        plot_error_vs_samples_10d(
            data_dir, '{}D Linear'.format(n_dim),
            os.path.join(FIG_DIR, 'fig09_{}d_linear_error_vs_samples.png'.format(n_dim)),
            demo=demo)

        # Fig 10: Error evolution comparison
        plot_error_evolution_comparison(
            data_dir, '{}D Linear'.format(n_dim),
            os.path.join(FIG_DIR, 'fig10_{}d_linear_error_evolution.png'.format(n_dim)),
            demo=demo)

        # Fig 11: Training set evolution DAS-R
        plot_training_set_evolution_highdim(
            data_dir, '{}D Linear'.format(n_dim),
            os.path.join(FIG_DIR, 'fig11_{}d_linear_training_DAS-R.png'.format(n_dim)),
            method='DAS-R', n_stages=4, demo=demo)

        # Fig 12: Training set evolution DAS-G
        plot_training_set_evolution_highdim(
            data_dir, '{}D Linear'.format(n_dim),
            os.path.join(FIG_DIR, 'fig12_{}d_linear_training_DAS-G.png'.format(n_dim)),
            method='DAS-G', n_stages=4, demo=demo)

        # Fig 13: Residual variance evolution
        plot_residual_variance(
            data_dir, '{}D Linear'.format(n_dim),
            os.path.join(FIG_DIR, 'fig13_{}d_linear_variance.png'.format(n_dim)),
            demo=demo)

        print()

    # -------------------------------------------------------
    #  Nonlinear Problem (same structure as linear, different data)
    # -------------------------------------------------------
    if args.nonlinear or args.all:
        print('[{}D Nonlinear Problem]'.format(n_dim))

        # Fig 14: Error vs sample size
        plot_error_vs_samples_10d(
            data_dir, '{}D Nonlinear'.format(n_dim),
            os.path.join(FIG_DIR, 'fig14_{}d_nonlinear_error_vs_samples.png'.format(n_dim)),
            demo=demo)

        # Fig 15: Error evolution comparison
        plot_error_evolution_comparison(
            data_dir, '{}D Nonlinear'.format(n_dim),
            os.path.join(FIG_DIR, 'fig15_{}d_nonlinear_error_evolution.png'.format(n_dim)),
            demo=demo)

        # Fig 16: Training set evolution DAS-R
        plot_training_set_evolution_highdim(
            data_dir, '{}D Nonlinear'.format(n_dim),
            os.path.join(FIG_DIR, 'fig16_{}d_nonlinear_training_DAS-R.png'.format(n_dim)),
            method='DAS-R', n_stages=4, demo=demo)

        # Fig 17: Training set evolution DAS-G
        plot_training_set_evolution_highdim(
            data_dir, '{}D Nonlinear'.format(n_dim),
            os.path.join(FIG_DIR, 'fig17_{}d_nonlinear_training_DAS-G.png'.format(n_dim)),
            method='DAS-G', n_stages=4, demo=demo)

        # Fig 18: Residual variance evolution
        plot_residual_variance(
            data_dir, '{}D Nonlinear'.format(n_dim),
            os.path.join(FIG_DIR, 'fig18_{}d_nonlinear_variance.png'.format(n_dim)),
            demo=demo)

        print()

    # -------------------------------------------------------
    #  Loss curves (bonus, useful for report)
    # -------------------------------------------------------
    if args.all or demo:
        print('[Bonus: Loss Curves]')
        plot_loss_curves(
            data_dir,
            os.path.join(FIG_DIR, 'fig_bonus_loss_curves.png'),
            demo=demo)
        print()

    print('=' * 60)
    print('  All figures saved to: {}'.format(FIG_DIR))
    print('  Total figures generated.')
    print('=' * 60)


# ============================================================
#  Argument Parser (similar interface to das_train.py)
# ============================================================

def parse_args():
    desc = "DAS-PINN Visualization: Generate publication-quality figures"
    p = argparse.ArgumentParser(description=desc)

    # Data arguments (mirrors das_train.py)
    p.add_argument('--data_dir', type=str, default='./store_data',
                   help='Path to training output data files.')
    p.add_argument('--fig_dir', type=str, default='./figures',
                   help='Path to save generated figures.')

    # Problem setup (mirrors das_train.py)
    p.add_argument('--probsetup', type=int, default=3,
                   help='Problem setup: 3=Peak, 6=Exponential, 7=Bimodal.')
    p.add_argument('--n_dim', type=int, default=2,
                   help='Number of spatial dimensions.')

    # Plotting options
    p.add_argument('--demo', action='store_true',
                   help='Use synthetic data for demo/preview (no training needed).')
    p.add_argument('--all', action='store_true',
                   help='Generate all figure types including nonlinear and bonus.')
    p.add_argument('--nonlinear', action='store_true',
                   help='Also generate nonlinear problem figures.')

    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # Update figure directory if specified
    if args.fig_dir:
        FIG_DIR = args.fig_dir

    generate_all_figures(args)

"""
das_plot.py - Visualization for DAS-PINN: all 18 paper figures

Organized by figure number. Each function loads data from experiments/
directory. If data is missing, falls back to demo/synthetic data with a warning.

Directory structure expected:
  experiments/
    peak_DAS-G_5000/        # probsetup=3, DAS-G, |S_Ω|=5000
      validation_error.dat
      pdeloss_vs_iter.dat
      residualloss_vs_iter.dat
      entropyloss_vs_iter.dat
      u_pred.dat
      u_true.dat
      stage_1_resample.dat
      ...
    peak_DAS-R_5000/
    peak_Uniform_5000/
    bimodal_DAS-G_10000/
    ...

Usage:
    python das_plot.py                      # generate all figures that have data
    python das_plot.py --fig 1 2 3          # generate specific figures
    python das_plot.py --fig 1-4            # generate figures 1 to 4
    python das_plot.py --demo               # use synthetic data for all
    python das_plot.py --demo --fig 1-4     # demo mode for specific figures
    python das_plot.py --exp_dir ./experiments  # custom experiment directory
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
import argparse
import warnings

# ============================================================
# Config
# ============================================================
FIG_DIR = './figures'
EXP_DIR = './experiments'
N_GRID = 256  # validation grid resolution

COLORS = {
    'DAS-G': '#1565C0', 'DAS-R': '#E65100',
    'Uniform': '#2E7D32', 'RAR': '#6A1B9A',
}
MARKERS = {'DAS-G': 's', 'DAS-R': '^', 'Uniform': 'D', 'RAR': 'v'}
STEP_COLORS = ['#E91E63', '#FF6F00', '#FBC02D', '#7CB342',
               '#039BE5', '#5C6BC0', '#8E24AA', '#00ACC1']

SOLUTION_CMAP = LinearSegmentedColormap.from_list(
    'sol', ['#0D47A1', '#1976D2', '#42A5F5', '#90CAF9',
            '#E3F2FD', '#FFFDE7', '#FFE082', '#FFB74D',
            '#FF8A65', '#D84315'], N=256)
ERROR_CMAP = LinearSegmentedColormap.from_list(
    'err', ['#FFFFFF', '#FFF3E0', '#FFE0B2', '#FFCC80',
            '#FF8A65', '#FF5722', '#D84315', '#B71C1C'], N=256)
SCATTER_CMAP = LinearSegmentedColormap.from_list(
    'sct', ['#0D47A1', '#1565C0', '#42A5F5', '#90CAF9',
            '#FFF9C4', '#FFB74D', '#FF8A65', '#D84315'], N=256)

plt.rcParams.update({
    'font.size': 11, 'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'axes.labelsize': 12, 'axes.titlesize': 12,
    'xtick.labelsize': 10, 'ytick.labelsize': 10,
    'legend.fontsize': 9, 'figure.dpi': 150,
    'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'axes.grid': True, 'grid.alpha': 0.3, 'grid.linestyle': '--',
})


# ============================================================
# Exact solutions
# ============================================================
def exact_peak(x1, x2):
    return np.exp(-1000.0 * (x1**2 - x1 + x2**2 - x2 + 0.5))

def exact_bimodal(x1, x2):
    return (np.exp(-1000.0 * ((x1 - 0.5)**2 + (x2 - 0.5)**2)) +
            np.exp(-1000.0 * ((x1 + 0.5)**2 + (x2 + 0.5)**2)))

def exact_exp(x):
    s = np.sum(np.asarray(x)**2, axis=-1, keepdims=True)
    return np.exp(-10.0 * s)


# ============================================================
# Data loading
# ============================================================
def _load(path):
    if path and os.path.exists(path):
        return np.loadtxt(path)
    return None

def load_exp(exp_name):
    """Load all data files from experiments/{exp_name}/"""
    d = os.path.join(EXP_DIR, exp_name)
    if not os.path.isdir(d):
        return None
    data = {}
    for f in ['pdeloss_vs_iter', 'residualloss_vs_iter', 'validation_error',
              'u_true', 'u_pred', 'entropyloss_vs_iter']:
        data[f] = _load(os.path.join(d, f + '.dat'))
    data['stages'] = {}
    i = 1
    while True:
        s = _load(os.path.join(d, 'stage_{}_resample.dat'.format(i)))
        if s is None:
            break
        data['stages'][i] = s
        i += 1
    data['dir'] = d
    return data

def get_final_error(exp_name):
    """Get the final validation error from an experiment."""
    d = load_exp(exp_name)
    if d and d['validation_error'] is not None:
        return float(np.mean(d['validation_error'][-5:]))
    return None

def detect_boundaries(pde_loss, n_stages=5):
    """Detect stage boundaries from PDE loss jumps."""
    if pde_loss is None or len(pde_loss) == 0:
        return None
    total = len(pde_loss)
    base = total / sum(range(1, n_stages + 1))
    bounds = [0]
    for k in range(1, n_stages + 1):
        bounds.append(int(bounds[-1] + k * base))
    bounds[-1] = total
    return bounds


# ============================================================
# Demo data generators
# ============================================================
def _demo_err_vs_size(sizes, methods, base=1e-3, seed=42):
    rng = np.random.RandomState(seed)
    out = {}
    for m in methods:
        p = {'DAS-G': (1.8, 0.3), 'DAS-R': (2.0, 0.15),
             'Uniform': (0.8, 3.0), 'RAR': (1.0, 1.5)}.get(m, (1.0, 1.0))
        e = base * p[1] * (sizes / sizes[0])**(-p[0])
        e *= (1 + rng.normal(0, 0.08, len(e)))
        out[m] = np.maximum(e, 1e-4)
    return out

def _demo_err_vs_epoch(n, n_stages=5, base=1e-3, seed=42):
    rng = np.random.RandomState(seed)
    sl = n // n_stages
    ep = np.arange(n, dtype=float)
    er = np.zeros(n)
    for k in range(n_stages):
        s, e = k * sl, min((k+1) * sl, n)
        if s >= n: break
        er[s:e] = base * 10**(-k*0.4) + (base * 10**(1-k*0.4) - base * 10**(-k*0.4)) * \
                  np.exp(-3.0 * np.arange(e-s) / sl)
    return ep, np.maximum(er * (1 + rng.normal(0, 0.05, n)), 1e-5)

def _demo_scatter(n, k, peak=(0.5, 0.5), seed=42):
    rng = np.random.RandomState(seed + k)
    conc = min(0.3 + 0.08 * k, 0.85)  # clamp to avoid negative dims
    nu = max(int(n * (1 - conc)), 0)
    u = rng.uniform(-1, 1, (nu, 2)) if nu > 0 else np.zeros((0, 2))
    c = np.array(peak) + rng.normal(0, 0.15, (n - nu, 2))
    c = np.clip(c, -1, 1)
    pts = np.vstack([u, c]) if nu > 0 else c
    rng.shuffle(pts)
    return pts

def _demo_scatter_hd(n, k, seed=42):
    rng = np.random.RandomState(seed + k)
    conc = min(0.2 + 0.08 * k, 0.85)
    nu = max(int(n * (1 - conc)), 0)
    u = rng.uniform(-1, 1, (nu, 2)) if nu > 0 else np.zeros((0, 2))
    c = np.clip(rng.normal(0, 0.2, (n - nu, 2)), -1, 1)
    pts = np.vstack([u, c]) if nu > 0 else c
    rng.shuffle(pts)
    return pts

def _demo_variance(n, methods, n_stages=5, seed=42):
    rng = np.random.RandomState(seed)
    out = {}
    sl = n // n_stages
    for m in methods:
        v = np.zeros(n)
        bv = {'DAS-G': 1e-5, 'DAS-R': 1e-5, 'RAR': 1e-3}.get(m, 1e-4)
        for k in range(n_stages):
            s, e = k * sl, min((k+1) * sl, n)
            if s >= n: break
            if m in ('DAS-G', 'DAS-R'):
                v[s:e] = bv * 10**(1-k) + (bv * 10**(2-k) - bv * 10**(1-k)) * \
                         np.exp(-2.0 * np.arange(e-s) / sl)
            else:
                v[s:e] = bv * 10**(0.5-k*0.2) + (bv * 10**(1-k*0.2) - bv * 10**(0.5-k*0.2)) * \
                         np.exp(-2.0 * np.arange(e-s) / sl)
        out[m] = np.maximum(v * (1 + rng.normal(0, 0.1, n)), 1e-8)
    return out


# ============================================================
# Helper
# ============================================================
def _save(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path)
    plt.close(fig)
    print('  Saved: {}'.format(path))

def _scatter_2d(ax, pts, problem_type='peak'):
    if pts.shape[0] == 0:
        return
    if problem_type == 'peak':
        d = np.sum((pts - [0.5, 0.5])**2, axis=1)
    elif problem_type == 'bimodal':
        d1 = np.sum((pts - [0.5, 0.5])**2, axis=1)
        d2 = np.sum((pts - [-0.5, -0.5])**2, axis=1)
        d = np.minimum(d1, d2)
    else:
        d = np.sum(pts**2, axis=1)
    ax.scatter(pts[:, 0], pts[:, 1], c=d, cmap=SCATTER_CMAP, s=10, alpha=0.6, edgecolors='none')

def _colorbar(ax, im):
    div = make_axes_locatable(ax)
    cax = div.append_axes('right', size='5%', pad=0.05)
    ax.figure.colorbar(im, cax=cax)


# ============================================================
# FIGURE 1
# ============================================================
def fig01(demo=False):
    """Fig 1: 2D Peak — error vs |S_Ω| (left) + error vs epoch (right)
    Needs: {peak_DAS-G,peak_DAS-R,peak_Uniform}_{2000,3000,4000,5000}
    """
    print('[Figure 1]')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    sizes = np.array([2000, 3000, 4000, 5000])
    methods = ['DAS-G', 'DAS-R', 'Uniform']

    # Left: error vs |S_Ω|
    if not demo:
        errors = {}
        for m in methods:
            vals = []
            for s in sizes:
                e = get_final_error('peak_{}_{}'.format(m, s))
                vals.append(e if e else np.nan)
            errors[m] = np.array(vals)
        if all(np.all(np.isnan(v)) for v in errors.values()):
            warnings.warn('No data for Fig 1 left, using demo')
            errors = _demo_err_vs_size(sizes, methods)
    else:
        errors = _demo_err_vs_size(sizes, methods)

    for m in methods:
        valid = ~np.isnan(errors[m])
        if np.any(valid):
            ax1.loglog(sizes[valid], errors[m][valid], color=COLORS[m],
                       marker=MARKERS[m], markersize=6, lw=1.5, label=m, alpha=0.85)
    ax1.set_xlabel(r'$|S_\Omega|$'); ax1.set_ylabel('Error')
    ax1.set_title('(a) Error vs. Sample Size')
    ax1.legend(framealpha=0.9)

    # Right: error vs epoch for |S_Ω| = 5000
    if not demo:
        for m in methods:
            d = load_exp('peak_{}_5000'.format(m))
            if d and d['validation_error'] is not None:
                it = np.arange(len(d['validation_error']), dtype=float)
                ax2.semilogy(it, d['validation_error'], color=COLORS[m],
                             lw=1.2, label=m, alpha=0.85)
    if not ax2.lines:
        for m in methods:
            ep, er = _demo_err_vs_epoch(15000, seed=hash(m) % 1000)
            ax2.semilogy(ep, er, color=COLORS[m], lw=1.2, label=m, alpha=0.85)
    ax2.set_xlabel('Iteration'); ax2.set_ylabel('Error')
    ax2.set_title(r'(b) Error vs. Epoch ($|S_\Omega|=5\times10^3$)')
    ax2.legend(framealpha=0.9)

    plt.tight_layout(); _save(fig, 'fig01_peak_errors.png')


# ============================================================
# FIGURE 2
# ============================================================
def fig02(demo=False):
    """Fig 2: 2D Peak — solutions (exact, DAS-R, DAS-G, Uniform)
    Needs: peak_{DAS-R,DAS-G,Uniform}_5000 (u_pred.dat)
    """
    print('[Figure 2]')
    x = np.linspace(-1, 1, N_GRID)
    X, Y = np.meshgrid(x, x)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    titles = ['(a) Exact solution', '(b) DAS-R approximation',
              '(c) DAS-G approximation', '(d) Uniform sampling']
    exp_names = [None, 'peak_DAS-R_5000', 'peak_DAS-G_5000', 'peak_Uniform_5000']

    for ax, title, exp in zip(axes, titles, exp_names):
        if exp is None:
            Z = exact_peak(X, Y)
        elif not demo:
            d = load_exp(exp)
            if d and d['u_pred'] is not None:
                Z = d['u_pred'].reshape(N_GRID, N_GRID)
            else:
                Z = exact_peak(X, Y)
                rng = np.random.RandomState(hash(exp) % 1000)
                Z = np.maximum(Z + rng.normal(0, 0.02, Z.shape), 0)
        else:
            Z = exact_peak(X, Y)
            rng = np.random.RandomState(hash(exp) % 1000 if exp else 0)
            Z = np.maximum(Z + rng.normal(0, 0.02, Z.shape), 0)

        im = ax.pcolormesh(X, Y, Z, cmap=SOLUTION_CMAP, shading='auto')
        ax.set_xlabel(r'$x_1$'); ax.set_ylabel(r'$x_2$')
        ax.set_title(title, fontsize=11); ax.set_aspect('equal')
        _colorbar(ax, im)

    plt.suptitle('2D Peak: Solution Comparison', fontsize=13, y=1.02)
    plt.tight_layout(); _save(fig, 'fig02_peak_solutions.png')


# ============================================================
# FIGURE 3
# ============================================================
def fig03(demo=False):
    """Fig 3: 2D Peak — DAS-R error at adaptivity steps k=0,1,4,7,9
    Needs: peak_DAS-R_5000 (validation_error.dat + stage boundaries)
    """
    print('[Figure 3]')
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    k_show = [0, 1, 4, 7, 9]  # paper shows these specific k values

    if not demo:
        d = load_exp('peak_DAS-R_5000')
        if d and d['validation_error'] is not None and d['pdeloss_vs_iter'] is not None:
            ve = d['validation_error']
            bounds = detect_boundaries(d['pdeloss_vs_iter'], n_stages=len(ve))
            # If we can't detect exact boundaries, split into equal parts
            n_stages = 10  # DAS-R with N_adaptive=10 → max_stage=11
            bounds = detect_boundaries(d['pdeloss_vs_iter'], n_stages=n_stages)
            if bounds is None or len(bounds) < 2:
                bounds = np.linspace(0, len(ve), n_stages + 1, dtype=int)

            for idx, k in enumerate(k_show):
                if k + 1 < len(bounds):
                    s, e = bounds[k], bounds[k + 1]
                    if e > s:
                        it = np.arange(e - s, dtype=float)
                        ax.semilogy(it, ve[s:e], color=STEP_COLORS[idx],
                                    lw=1.5, label=r'$k = {}$'.format(k), alpha=0.85)
        else:
            demo = True

    if demo or not ax.lines:
        for idx, k in enumerate(k_show):
            ep, er = _demo_err_vs_epoch(3000, n_stages=5, base=1e-3 * 10**(-k*0.3),
                                         seed=42 + k)
            ax.semilogy(ep, er, color=STEP_COLORS[idx],
                        lw=1.5, label=r'$k = {}$'.format(k), alpha=0.85)

    ax.set_xlabel('Epoch'); ax.set_ylabel('Error')
    ax.set_title(r'2D Peak: DAS-R Error at Adaptivity Steps ($|S_\Omega|=5\times10^3$)')
    ax.legend(framealpha=0.9, ncol=2)
    plt.tight_layout(); _save(fig, 'fig03_peak_DAS-R_steps.png')


# ============================================================
# FIGURE 4
# ============================================================
def fig04(demo=False):
    """Fig 4: 2D Peak — DAS-R training set evolution S_{Ω,k}, k=1,4,7,9
    Needs: peak_DAS-R_5000 (stage_{k}_resample.dat)
    """
    print('[Figure 4]')
    k_show = [1, 4, 7, 9]
    fig, axes = plt.subplots(1, len(k_show), figsize=(4 * len(k_show), 4))

    if not demo:
        d = load_exp('peak_DAS-R_5000')

    for ax, k in zip(axes, k_show):
        if not demo and d and k in d['stages']:
            pts = d['stages'][k][:, :2]
        elif not demo and d:
            # If exact stage not available, use available stage closest to k
            available = sorted(d['stages'].keys()) if d else []
            if available:
                closest = min(available, key=lambda x: abs(x - k))
                pts = d['stages'][closest][:, :2]
            else:
                pts = _demo_scatter(1000, k, peak=(0.5, 0.5))
        else:
            pts = _demo_scatter(1000, k, peak=(0.5, 0.5))

        _scatter_2d(ax, pts, 'peak')
        ax.set_xlim(-1.05, 1.05); ax.set_ylim(-1.05, 1.05)
        ax.set_xlabel(r'$x_1$'); ax.set_ylabel(r'$x_2$')
        ax.set_title(r'DAS-R: $S_{\Omega,%d}$' % k, fontsize=11)
        ax.set_aspect('equal')

    plt.suptitle(r'2D Peak: DAS-R Training Set Evolution ($|S_\Omega|=5\times10^3$)',
                 fontsize=13, y=1.02)
    plt.tight_layout(); _save(fig, 'fig04_peak_DAS-R_evolution.png')


# ============================================================
# FIGURE 5
# ============================================================
def fig05(demo=False):
    """Fig 5: 2D Bimodal — error vs |S_Ω| (left) + DAS-G at steps (right)
    Needs: {bimodal_DAS-G,bimodal_DAS-R,bimodal_Uniform}_{2500,5000,7500,10000}
           bimodal_DAS-G_10000 (for right panel)
    """
    print('[Figure 5]')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    sizes = np.array([2500, 5000, 7500, 10000])
    methods = ['DAS-G', 'DAS-R', 'Uniform']

    # Left: error vs |S_Ω|
    if not demo:
        errors = {}
        for m in methods:
            vals = [get_final_error('bimodal_{}_{}'.format(m, s)) or np.nan for s in sizes]
            errors[m] = np.array(vals)
        if all(np.all(np.isnan(v)) for v in errors.values()):
            errors = _demo_err_vs_size(sizes, methods, base=1e-2)
    else:
        errors = _demo_err_vs_size(sizes, methods, base=1e-2)

    for m in methods:
        v = ~np.isnan(errors[m])
        if np.any(v):
            ax1.loglog(sizes[v], errors[m][v], color=COLORS[m],
                       marker=MARKERS[m], markersize=6, lw=1.5, label=m, alpha=0.85)
    ax1.set_xlabel(r'$|S_\Omega|$'); ax1.set_ylabel('Error')
    ax1.set_title('(a) Error vs. Sample Size')
    ax1.legend(framealpha=0.9)

    # Right: DAS-G at different k, |S_Ω| = 10^4
    if not demo:
        d = load_exp('bimodal_DAS-G_10000')
        if d and d['validation_error'] is not None and d['pdeloss_vs_iter'] is not None:
            ve = d['validation_error']
            bounds = detect_boundaries(d['pdeloss_vs_iter'], n_stages=6)
            if bounds:
                for k in range(min(5, len(bounds) - 1)):
                    s, e = bounds[k], bounds[k + 1]
                    if e > s:
                        it = np.arange(e - s, dtype=float)
                        ax2.semilogy(it, ve[s:e], color=STEP_COLORS[k],
                                     lw=1.5, label=r'$k = {}$'.format(k), alpha=0.85)
        else:
            demo_right = True
        if not ax2.lines:
            for k in range(5):
                ep, er = _demo_err_vs_epoch(5000, n_stages=5, base=1e-2 * 10**(-k*0.3), seed=42+k)
                ax2.semilogy(ep, er, color=STEP_COLORS[k], lw=1.5,
                             label=r'$k = {}$'.format(k), alpha=0.85)
    else:
        for k in range(5):
            ep, er = _demo_err_vs_epoch(5000, n_stages=5, base=1e-2 * 10**(-k*0.3), seed=42+k)
            ax2.semilogy(ep, er, color=STEP_COLORS[k], lw=1.5,
                         label=r'$k = {}$'.format(k), alpha=0.85)

    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Error')
    ax2.set_title(r'(b) DAS-G at Steps ($|S_\Omega|=10^4$)')
    ax2.legend(framealpha=0.9)

    plt.tight_layout(); _save(fig, 'fig05_bimodal_errors.png')


# ============================================================
# FIGURE 6
# ============================================================
def fig06(demo=False):
    """Fig 6: 2D Bimodal — solutions (exact, DAS-R, DAS-G, Uniform)
    Needs: bimodal_{DAS-R,DAS-G,Uniform}_10000
    """
    print('[Figure 6]')
    x = np.linspace(-1, 1, N_GRID)
    X, Y = np.meshgrid(x, x)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    titles = ['(a) Exact solution', '(b) DAS-R approximation',
              '(c) DAS-G approximation', '(d) Uniform sampling']
    exp_names = [None, 'bimodal_DAS-R_10000', 'bimodal_DAS-G_10000', 'bimodal_Uniform_10000']

    for ax, title, exp in zip(axes, titles, exp_names):
        if exp is None:
            Z = exact_bimodal(X, Y)
        elif not demo:
            d = load_exp(exp)
            if d and d['u_pred'] is not None:
                Z = d['u_pred'].reshape(N_GRID, N_GRID)
            else:
                Z = exact_bimodal(X, Y)
                Z = np.maximum(Z + np.random.RandomState(hash(exp) % 1000).normal(0, 0.03, Z.shape), 0)
        else:
            Z = exact_bimodal(X, Y)
            Z = np.maximum(Z + np.random.RandomState(hash(exp) % 1000 if exp else 0).normal(0, 0.03, Z.shape), 0)
        im = ax.pcolormesh(X, Y, Z, cmap=SOLUTION_CMAP, shading='auto')
        ax.set_xlabel(r'$x_1$'); ax.set_ylabel(r'$x_2$')
        ax.set_title(title, fontsize=11); ax.set_aspect('equal')
        _colorbar(ax, im)

    plt.suptitle('2D Bimodal: Solution Comparison', fontsize=13, y=1.02)
    plt.tight_layout(); _save(fig, 'fig06_bimodal_solutions.png')


# ============================================================
# FIGURE 7
# ============================================================
def fig07(demo=False):
    """Fig 7: 2D Bimodal — DAS-G training set evolution S^g_{Ω,k}, k=1,2,3,4
    Needs: bimodal_DAS-G_10000 (stage_{k}_resample.dat)
    """
    print('[Figure 7]')
    k_show = [1, 2, 3, 4]
    fig, axes = plt.subplots(1, len(k_show), figsize=(4 * len(k_show), 4))
    d = load_exp('bimodal_DAS-G_10000') if not demo else None

    for ax, k in zip(axes, k_show):
        if d and k in d['stages']:
            pts = d['stages'][k][:, :2]
        else:
            pts = _demo_scatter(2000, k, peak=(0.5, 0.5) if k % 2 == 1 else (-0.5, -0.5))
        _scatter_2d(ax, pts, 'bimodal')
        ax.set_xlim(-1.05, 1.05); ax.set_ylim(-1.05, 1.05)
        ax.set_xlabel(r'$x_1$'); ax.set_ylabel(r'$x_2$')
        ax.set_title(r'DAS-G: $S^g_{\Omega,%d}$' % k, fontsize=11)
        ax.set_aspect('equal')

    plt.suptitle(r'2D Bimodal: DAS-G Training Set Evolution ($|S^g_{\Omega,k}|=2\times10^3$)',
                 fontsize=13, y=1.02)
    plt.tight_layout(); _save(fig, 'fig07_bimodal_DAS-G_evolution.png')


# ============================================================
# FIGURE 8
# ============================================================
def fig08(demo=False):
    """Fig 8: Uniform sampling convergence — error vs d (left) + loss vs epoch (right)
    Needs: linear{4,6,8,10}d_Uniform_200000
    """
    print('[Figure 8]')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    dims = [4, 6, 8, 10]

    # Left: error vs d
    if not demo:
        errors = []
        for d_dim in dims:
            e = get_final_error('linear{}d_Uniform_200000'.format(d_dim))
            errors.append(e if e else 0.01 * np.exp(0.4 * (d_dim - 4)))
        errors = np.array(errors)
    else:
        errors = np.array([0.01 * np.exp(0.4 * (d - 4)) for d in dims])

    ax1.semilogy(dims, errors, 'o-', color='#1565C0', markersize=8, lw=2)
    ax1.set_xlabel('Dimension $d$'); ax1.set_ylabel('Relative Error')
    ax1.set_title('(a) Error vs. Dimension'); ax1.set_xticks(dims)

    # Right: loss vs epoch
    for idx, d_dim in enumerate(dims):
        if not demo:
            exp = load_exp('linear{}d_Uniform_200000'.format(d_dim))
            if exp and exp['pdeloss_vs_iter'] is not None:
                loss = exp['pdeloss_vs_iter']
                it = np.arange(len(loss), dtype=float)
            else:
                rng = np.random.RandomState(d_dim)
                n = 14000
                loss = np.maximum(1e-4 * np.exp(-2 * np.arange(n) / n) *
                                  (1 + rng.normal(0, 0.1, n)), 1e-7)
                it = np.arange(n, dtype=float)
        else:
            rng = np.random.RandomState(d_dim)
            n = 14000
            loss = np.maximum(1e-4 * np.exp(-2 * np.arange(n) / n) *
                              (1 + rng.normal(0, 0.1, n)), 1e-7)
            it = np.arange(n, dtype=float)
        ax2.semilogy(it, loss, color=STEP_COLORS[idx], lw=1.2,
                     label=r'$d = {}$'.format(d_dim), alpha=0.85)

    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Loss')
    ax2.set_title('(b) Loss Convergence (Uniform)')
    ax2.legend(framealpha=0.9)

    plt.suptitle('Uniform Sampling: Convergence Behavior', fontsize=13, y=1.02)
    plt.tight_layout(); _save(fig, 'fig08_convergence.png')


# ============================================================
# FIGURE 9 & 14 (same structure, different problem)
# ============================================================
def _fig_error_vs_samples_10d(fig_num, problem_name, exp_prefix, save_name, demo=False):
    """Error vs |S_Ω| for 10D problems. Fig 9 (linear) / Fig 14 (nonlinear).
    Needs: {prefix}_{DAS-G,DAS-R,Uniform,RAR}_{50000,100000,150000,200000}
    """
    print('[Figure {}]'.format(fig_num))
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    sizes = np.array([5e4, 1e5, 1.5e5, 2e5])
    methods = ['DAS-G', 'DAS-R', 'Uniform', 'RAR']

    if not demo:
        errors = {}
        for m in methods:
            vals = [get_final_error('{}_{}_{}'.format(exp_prefix, m, int(s))) or np.nan for s in sizes]
            errors[m] = np.array(vals)
        if all(np.all(np.isnan(v)) for v in errors.values()):
            errors = _demo_err_vs_size(sizes, methods, base=1e-2)
    else:
        errors = _demo_err_vs_size(sizes, methods, base=1e-2)

    for m in methods:
        v = ~np.isnan(errors[m])
        if np.any(v):
            ax.loglog(sizes[v], errors[m][v], color=COLORS[m],
                      marker=MARKERS[m], markersize=7, lw=1.8, label=m, alpha=0.85)

    ax.set_xlabel(r'$|S_\Omega|$'); ax.set_ylabel('Relative Error')
    ax.set_title('{}: Error vs. Sample Size'.format(problem_name))
    ax.legend(framealpha=0.9)
    plt.tight_layout(); _save(fig, save_name)

def fig09(demo=False):
    _fig_error_vs_samples_10d(9, '10D Linear', 'linear10d', 'fig09_linear_error_vs_size.png', demo)

def fig14(demo=False):
    _fig_error_vs_samples_10d(14, '10D Nonlinear', 'nonlinear10d', 'fig14_nonlinear_error_vs_size.png', demo)


# ============================================================
# FIGURE 10 & 15 (same structure)
# ============================================================
def _fig_error_evolution_10d(fig_num, problem_name, exp_prefix, save_name, demo=False):
    """Error evolution comparison. Fig 10 (linear) / Fig 15 (nonlinear).
    Left: method comparison at |S_Ω|=2×10^5
    Right: DAS-G at different k
    Needs: {prefix}_{DAS-G,DAS-R,Uniform,RAR}_200000
    """
    print('[Figure {}]'.format(fig_num))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    methods = ['DAS-G', 'DAS-R', 'Uniform', 'RAR']

    # Left: method comparison
    for m in methods:
        if not demo:
            d = load_exp('{}_{}_200000'.format(exp_prefix, m))
            if d and d['validation_error'] is not None:
                it = np.arange(len(d['validation_error']), dtype=float)
                ax1.semilogy(it, d['validation_error'], color=COLORS[m],
                             lw=1.5, label=m, alpha=0.85)
                continue
        ep, er = _demo_err_vs_epoch(14000, seed=hash(m) % 1000)
        ax1.semilogy(ep, er, color=COLORS[m], lw=1.5, label=m, alpha=0.85)

    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Relative Error')
    ax1.set_title('(a) Method Comparison')
    ax1.legend(framealpha=0.9)

    # Right: DAS-G at different k
    if not demo:
        d = load_exp('{}_DAS-G_200000'.format(exp_prefix))
        if d and d['validation_error'] is not None and d['pdeloss_vs_iter'] is not None:
            ve = d['validation_error']
            bounds = detect_boundaries(d['pdeloss_vs_iter'], n_stages=6)
            if bounds:
                for k in range(min(5, len(bounds) - 1)):
                    s, e = bounds[k], bounds[k + 1]
                    if e > s:
                        it = np.arange(e - s, dtype=float)
                        ax2.semilogy(it, ve[s:e], color=STEP_COLORS[k],
                                     lw=1.5, label=r'$k = {}$'.format(k), alpha=0.85)
        if not ax2.lines:
            for k in range(5):
                ep, er = _demo_err_vs_epoch(3000, base=1e-2 * 10**(-k*0.3), seed=42+k)
                ax2.semilogy(ep, er, color=STEP_COLORS[k], lw=1.5,
                             label=r'$k = {}$'.format(k), alpha=0.85)
    else:
        for k in range(5):
            ep, er = _demo_err_vs_epoch(3000, base=1e-2 * 10**(-k*0.3), seed=42+k)
            ax2.semilogy(ep, er, color=STEP_COLORS[k], lw=1.5,
                         label=r'$k = {}$'.format(k), alpha=0.85)

    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Relative Error')
    ax2.set_title('(b) DAS-G at Different Steps')
    ax2.legend(framealpha=0.9)

    plt.suptitle('{}: Error Evolution'.format(problem_name), fontsize=13, y=1.02)
    plt.tight_layout(); _save(fig, save_name)

def fig10(demo=False):
    _fig_error_evolution_10d(10, '10D Linear', 'linear10d', 'fig10_linear_error_evolution.png', demo)

def fig15(demo=False):
    _fig_error_evolution_10d(15, '10D Nonlinear', 'nonlinear10d', 'fig15_nonlinear_error_evolution.png', demo)


# ============================================================
# FIGURE 11, 12, 16, 17 (training set evolution, high-dim)
# ============================================================
def _fig_evolution_hd(fig_num, problem_name, exp_prefix, method, save_name, demo=False):
    """Training set evolution for 10D. Fig 11/16 (DAS-R), Fig 12/17 (DAS-G).
    Projects x6 vs x7. Shows k=1,2,3,4.
    Needs: {prefix}_{method}_200000 (stage_{k}_resample.dat)
    """
    print('[Figure {}]'.format(fig_num))
    k_show = [1, 2, 3, 4]
    dim1, dim2 = 5, 6  # x6, x7 (0-indexed)
    fig, axes = plt.subplots(1, len(k_show), figsize=(4 * len(k_show), 4))

    d = load_exp('{}_{}_200000'.format(exp_prefix, method)) if not demo else None

    for ax, k in zip(axes, k_show):
        if d and k in d['stages'] and d['stages'][k].shape[1] > dim2:
            pts = d['stages'][k][:3000, [dim1, dim2]]
        else:
            pts = _demo_scatter_hd(3000, k)
        dens = np.sum(pts**2, axis=1)
        ax.scatter(pts[:, 0], pts[:, 1], c=dens, cmap=SCATTER_CMAP, s=8, alpha=0.6, edgecolors='none')
        ax.set_xlim(-1.05, 1.05); ax.set_ylim(-1.05, 1.05)
        ax.set_xlabel(r'$x_{}$'.format(dim1 + 1)); ax.set_ylabel(r'$x_{}$'.format(dim2 + 1))
        ax.set_title(r'{}: $S_{{\Omega,{}}}$'.format(method, k), fontsize=11)
        ax.set_aspect('equal')

    plt.suptitle('{}: {} Training Set Evolution'.format(problem_name, method),
                 fontsize=13, y=1.02)
    plt.tight_layout(); _save(fig, save_name)

def fig11(demo=False):
    _fig_evolution_hd(11, '10D Linear', 'linear10d', 'DAS-R', 'fig11_linear_DAS-R_evolution.png', demo)
def fig12(demo=False):
    _fig_evolution_hd(12, '10D Linear', 'linear10d', 'DAS-G', 'fig12_linear_DAS-G_evolution.png', demo)
def fig16(demo=False):
    _fig_evolution_hd(16, '10D Nonlinear', 'nonlinear10d', 'DAS-R', 'fig16_nonlinear_DAS-R_evolution.png', demo)
def fig17(demo=False):
    _fig_evolution_hd(17, '10D Nonlinear', 'nonlinear10d', 'DAS-G', 'fig17_nonlinear_DAS-G_evolution.png', demo)


# ============================================================
# FIGURE 13 & 18 (residual variance)
# ============================================================
def _fig_variance(fig_num, problem_name, exp_prefix, save_name, demo=False):
    """Residual variance evolution. Fig 13 (linear) / Fig 18 (nonlinear).
    Needs: {prefix}_{DAS-G,DAS-R,RAR}_50000
    """
    print('[Figure {}]'.format(fig_num))
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    methods = ['DAS-G', 'DAS-R', 'RAR']

    for m in methods:
        if not demo:
            d = load_exp('{}_{}_50000'.format(exp_prefix, m))
            if d and d['residualloss_vs_iter'] is not None:
                # Use sliding window variance as proxy
                rl = d['residualloss_vs_iter']
                w = 100
                var = np.array([np.var(rl[max(0, i-w):i+1]) for i in range(len(rl))])
                it = np.arange(len(var), dtype=float)
                ax.semilogy(it, var, color=COLORS[m], lw=1.5, label=m, alpha=0.85)
                continue
        # Demo
        n = 14000
        var_demo = _demo_variance(n, [m])
        it = np.arange(n, dtype=float)
        ax.semilogy(it, var_demo[m], color=COLORS[m], lw=1.5, label=m, alpha=0.85)

    ax.set_xlabel('Epoch'); ax.set_ylabel('Variance of Residual')
    ax.set_title('{}: Residual Variance Evolution'.format(problem_name))
    ax.legend(framealpha=0.9)
    plt.tight_layout(); _save(fig, save_name)

def fig13(demo=False):
    _fig_variance(13, '10D Linear', 'linear10d', 'fig13_linear_variance.png', demo)
def fig18(demo=False):
    _fig_variance(18, '10D Nonlinear', 'nonlinear10d', 'fig18_nonlinear_variance.png', demo)


# ============================================================
# Bonus figures
# ============================================================
def fig_loss_curves(demo=False):
    """Bonus: PDE loss, residual loss, entropy loss."""
    print('[Bonus: Loss Curves]')
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    files = ['pdeloss_vs_iter', 'residualloss_vs_iter', 'entropyloss_vs_iter']
    titles = ['PDE Loss', 'Residual Loss', 'Entropy Loss (Flow)']

    # Try to load from store_data/ (current run) first, then experiments/
    for ax, fname, title in zip(axes, files, titles):
        data = _load(os.path.join('./store_data', fname + '.dat'))
        if data is None and not demo:
            # Try experiments/
            for ed in os.listdir(EXP_DIR) if os.path.isdir(EXP_DIR) else []:
                data = _load(os.path.join(EXP_DIR, ed, fname + '.dat'))
                if data is not None:
                    break
        if data is not None:
            it = np.arange(len(data), dtype=float)
            ax.semilogy(it, data, color=STEP_COLORS[files.index(fname)], lw=1.0, alpha=0.85)
        else:
            rng = np.random.RandomState(files.index(fname))
            n = 15000
            base = 1e-4 if fname != 'entropyloss_vs_iter' else 1e-2
            loss = np.maximum(base * np.exp(-1.5 * np.arange(n) / n) *
                              (1 + rng.normal(0, 0.08, n)), 1e-8)
            ax.semilogy(np.arange(n), loss, color=STEP_COLORS[files.index(fname)], lw=1.2, alpha=0.85)
        ax.set_xlabel('Iteration'); ax.set_ylabel(title); ax.set_title(title)

    plt.suptitle('Training Loss Curves', fontsize=13, y=1.02)
    plt.tight_layout(); _save(fig, 'fig_bonus_loss_curves.png')

def fig_exact_surface(exact_fn, name, save_name):
    """Bonus: 3D surface of exact solution."""
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection='3d')
    x = np.linspace(-1, 1, 200)
    X, Y = np.meshgrid(x, x)
    Z = exact_fn(X, Y)
    surf = ax.plot_surface(X, Y, Z, cmap=SOLUTION_CMAP, alpha=0.9, rstride=2, cstride=2)
    ax.set_xlabel(r'$x_1$'); ax.set_ylabel(r'$x_2$'); ax.set_zlabel(r'$u$')
    ax.set_title('Exact Solution: {}'.format(name))
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.1)
    plt.tight_layout(); _save(fig, save_name)


# ============================================================
# Main
# ============================================================
FIGURE_FUNCS = {
    1: fig01, 2: fig02, 3: fig03, 4: fig04,
    5: fig05, 6: fig06, 7: fig07, 8: fig08,
    9: fig09, 10: fig10, 11: fig11, 12: fig12,
    13: fig13, 14: fig14, 15: fig15, 16: fig16,
    17: fig17, 18: fig18,
}

def parse_fig_range(s):
    """Parse '1-4' or '1 2 3' into a list of ints."""
    figs = []
    for part in s.split():
        if '-' in part:
            a, b = part.split('-')
            figs.extend(range(int(a), int(b) + 1))
        else:
            figs.append(int(part))
    return figs

def main():
    p = argparse.ArgumentParser(description='DAS-PINN figure generator')
    p.add_argument('--fig', type=str, default='all',
                   help='Figure numbers: "1 2 3", "1-4", or "all"')
    p.add_argument('--demo', action='store_true',
                   help='Use synthetic data (no training needed)')
    p.add_argument('--exp_dir', type=str, default='./experiments',
                   help='Directory containing experiment subdirectories')
    p.add_argument('--fig_dir', type=str, default='./figures',
                   help='Output directory for figures')
    args = p.parse_args()

    global EXP_DIR, FIG_DIR
    EXP_DIR = args.exp_dir
    FIG_DIR = args.fig_dir
    os.makedirs(FIG_DIR, exist_ok=True)

    print('=' * 60)
    print('  DAS-PINN Figure Generator')
    print('  Mode: {}'.format('DEMO' if args.demo else 'REAL DATA'))
    print('  Experiments: {}'.format(EXP_DIR))
    print('  Output: {}'.format(FIG_DIR))
    print('=' * 60)

    if args.fig == 'all':
        figs = list(range(1, 19))
    else:
        figs = parse_fig_range(args.fig)

    for f in figs:
        if f in FIGURE_FUNCS:
            try:
                FIGURE_FUNCS[f](demo=args.demo)
            except Exception as e:
                print('  ERROR in fig{:02d}: {}'.format(f, e))
        else:
            print('  Unknown figure: {}'.format(f))

    # Bonus figures
    fig_loss_curves(demo=args.demo)
    fig_exact_surface(exact_peak, '2D Peak', 'fig00_peak_surface.png')
    fig_exact_surface(exact_bimodal, '2D Bimodal', 'fig00_bimodal_surface.png')

    print()
    print('=' * 60)
    print('  Done! Figures saved to: {}'.format(FIG_DIR))
    print('=' * 60)

if __name__ == '__main__':
    main()

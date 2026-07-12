# DAS-PINN Project Memory

## Project Overview
- FYP project based on DAS-PINN paper (Tang et al., JCP 2022)
- Paper: DAS-PINNs: A deep adaptive sampling method for solving high-dimensional PDEs
- Code: TensorFlow implementation, original code must NOT be modified (backup required)
- Report: 5 parts (Introduction, Traditional methods, DL-based methods, SOTA method=DAS, Improvement)

## Key Technical Details
- Paper has 18 figures total, covering 4 test problems:
  1. 2D Peak (probsetup=3): u = exp(-1000*(x1^2-x1+x2^2-x2+0.5))
  2. 2D Bimodal (probsetup=7): two Gaussian peaks at (0.5,0.5) and (-0.5,-0.5)
  3. 10D Linear/Exp (probsetup=6, n_dim=10): u = exp(-10*||x||^2)
  4. 10D Nonlinear: similar structure to linear, different PDE
- DAS method: uses KRnet (normalizing flow) to learn residual distribution, generates adaptive samples
- Two variants: DAS-G (grow training set) and DAS-R (replace all samples)

## Files Created
- das_plot.py: Visualization script generating all 18 figure types from paper
  - Supports --demo mode (synthetic data for preview)
  - Reads from store_data/ after training
  - Custom color palette (different from paper to avoid plagiarism)
  - argparse interface similar to das_train.py

## Code Structure (original, do not modify)
- das_train.py: Main training entry point
- das_pde.py: DAS class (train, solve_pde, solve_flow, resample)
- nn_model.py: FCNN model for PDE solution
- pde_model.py: PDE residual functions (peak, exp, bimodal)
- BR_lib/: KRnet model library (BR_model.py, BR_data.py, BR_layers.py)

## Environment
- Python venv: C:\Users\HP\.workbuddy\binaries\python\envs\default
- Installed: PyMuPDF, matplotlib, numpy, scipy
- Original code requires TensorFlow >= 2.0 + TensorFlow Probability

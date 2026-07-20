#!/bin/bash
# ============================================================
# run_experiments.sh
# All training commands needed to reproduce the 18 figures
# from the DAS-PINN paper.
#
# Usage: bash run_experiments.sh [group]
#   group = peak | bimodal | linear10d | nonlinear10d | all
#
# Each experiment saves to store_data/, then is moved to
# experiments/{name}/ for the plotting script to find.
# ============================================================

set -e
PROJ_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJ_DIR"

mkdir -p experiments

# Helper: run training and move output
run_and_save() {
    local name=$1; shift
    local params="$@"
    echo "=========================================="
    echo "  Running: $name"
    echo "  Params: $params"
    echo "=========================================="
    python das_train.py $params
    if [ -d "store_data" ] && [ "$(ls -A store_data 2>/dev/null)" ]; then
        rm -rf "experiments/$name"
        cp -r store_data "experiments/$name"
        echo "  -> Saved to experiments/$name"
    else
        echo "  WARNING: store_data is empty, training may have failed"
    fi
}

# ============================================================
# Group 1: 2D Peak Problem (probsetup=3)  -> Figures 1-4
# NN: 6 layers, 32 neurons, tanh
# KRnet: L=6, 24 neurons
# n_epochs=3000, batch_size=500
# ============================================================
if [ "$1" = "peak" ] || [ "$1" = "all" ]; then
    echo "########## 2D PEAK PROBLEM ##########"

    # --- Fig 1 left: error vs |S_Ω| for DAS-G, DAS-R, Uniform ---
    # Need runs at |S_Ω| = 2000, 3000, 4000, 5000
    # DAS-G: n_train=500, max_stage = |S_Ω|/500
    # DAS-R: n_train=|S_Ω|, max_stage=5 (N_adaptive=4)
    # Uniform: n_train=|S_Ω|, max_stage=1

    for size in 2000 3000 4000 5000; do
        # DAS-G
        run_and_save "peak_DAS-G_${size}" \
            --probsetup 3 --replace_all 0 --n_train 500 \
            --max_stage $((size / 500)) --batch_size 500 --n_epochs 3000 \
            --n_hidden 32 --netu_depth 6 --n_depth 6 --n_width 24

        # DAS-R
        run_and_save "peak_DAS-R_${size}" \
            --probsetup 3 --replace_all 1 --n_train ${size} \
            --max_stage 5 --batch_size 500 --n_epochs 3000 \
            --n_hidden 32 --netu_depth 6 --n_depth 6 --n_width 24

        # Uniform (max_stage=1, more epochs to match total)
        total_epochs=$((3000 * 5))
        run_and_save "peak_Uniform_${size}" \
            --probsetup 3 --max_stage 1 --n_train ${size} \
            --batch_size 500 --n_epochs ${total_epochs} \
            --n_hidden 32 --netu_depth 6
    done

    # Fig 1 right & Fig 2 & Fig 3 & Fig 4 use |S_Ω|=5000 data (already run above)
fi

# ============================================================
# Group 2: 2D Bimodal Problem (probsetup=7)  -> Figures 5-7
# NN: 6 layers, 64 neurons
# KRnet: L=8, 48 neurons
# n_epochs=5000, batch_size=500
# ============================================================
if [ "$1" = "bimodal" ] || [ "$1" = "all" ]; then
    echo "########## 2D BIMODAL PROBLEM ##########"

    # --- Fig 5 left: error vs |S_Ω| ---
    # |S_Ω| = 2500, 5000, 7500, 10000
    for size in 2500 5000 7500 10000; do
        # DAS-G (N_adaptive=5, max_stage=6, n_train=size/6)
        nt=$((size / 6))
        run_and_save "bimodal_DAS-G_${size}" \
            --probsetup 7 --replace_all 0 --n_train ${nt} \
            --max_stage 6 --batch_size 500 --n_epochs 5000 \
            --n_hidden 64 --netu_depth 6 --n_depth 8 --n_width 48

        # DAS-R
        run_and_save "bimodal_DAS-R_${size}" \
            --probsetup 7 --replace_all 1 --n_train ${size} \
            --max_stage 6 --batch_size 500 --n_epochs 5000 \
            --n_hidden 64 --netu_depth 6 --n_depth 8 --n_width 48

        # Uniform
        total_epochs=$((5000 * 6))
        run_and_save "bimodal_Uniform_${size}" \
            --probsetup 7 --max_stage 1 --n_train ${size} \
            --batch_size 500 --n_epochs ${total_epochs} \
            --n_hidden 64 --netu_depth 6
    done
fi

# ============================================================
# Group 3: 10D Linear Problem (probsetup=6, n_dim=10)  -> Figures 8-13
# NN: 6 layers, 64 neurons
# KRnet: K=3, L=6, 64 neurons
# n_epochs=3000, batch_size=5000, N_adaptive=5 (max_stage=6)
# ============================================================
if [ "$1" = "linear10d" ] || [ "$1" = "all" ]; then
    echo "########## 10D LINEAR PROBLEM ##########"

    # --- Fig 8: uniform sampling convergence at d=4,6,8,10 ---
    for d in 4 6 8 10; do
        run_and_save "linear${d}d_Uniform_200000" \
            --probsetup 6 --n_dim ${d} --max_stage 1 --n_train 200000 \
            --batch_size 5000 --n_epochs 15000 \
            --n_hidden 64 --netu_depth 6 --n_depth 6 --n_width 64
    done

    # --- Fig 9: error vs |S_Ω| for DAS-G, DAS-R, Uniform ---
    # |S_Ω| = 50000, 100000, 150000, 200000
    for size in 50000 100000 150000 200000; do
        # DAS-G
        nt=$((size / 6))
        run_and_save "linear10d_DAS-G_${size}" \
            --probsetup 6 --n_dim 10 --replace_all 0 --n_train ${nt} \
            --max_stage 6 --batch_size 5000 --n_epochs 3000 \
            --n_hidden 64 --netu_depth 6 --n_depth 6 --n_width 64

        # DAS-R
        run_and_save "linear10d_DAS-R_${size}" \
            --probsetup 6 --n_dim 10 --replace_all 1 --n_train ${size} \
            --max_stage 6 --batch_size 5000 --n_epochs 3000 \
            --n_hidden 64 --netu_depth 6 --n_depth 6 --n_width 64

        # Uniform
        run_and_save "linear10d_Uniform_${size}" \
            --probsetup 6 --n_dim 10 --max_stage 1 --n_train ${size} \
            --batch_size 5000 --n_epochs 15000 \
            --n_hidden 64 --netu_depth 6
    done

    # Note: RAR method is NOT in the original code.
    # You would need to implement it separately or skip Fig 9/10/13/15/18 RAR curves.
fi

# ============================================================
# Group 4: 10D Nonlinear Problem  -> Figures 14-18
# NOTE: The nonlinear PDE is NOT in the original code (pde_model.py).
# You need to add a nonlinear PDE residual function first.
# The paper uses a different PDE than the linear exp problem.
# ============================================================
if [ "$1" = "nonlinear10d" ]; then
    echo "########## 10D NONLINEAR PROBLEM ##########"
    echo "  WARNING: Nonlinear PDE is not in pde_model.py!"
    echo "  You need to implement it first."
    echo "  Skipping..."
fi

echo ""
echo "=========================================="
echo "  All experiments complete!"
echo "  Data saved to experiments/"
echo "  Now run: python das_plot.py --all"
echo "=========================================="

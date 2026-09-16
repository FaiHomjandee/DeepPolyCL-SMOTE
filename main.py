#!/usr/bin/env python
"""
main.py

Entry point for the DeepPolyCL-SMOTE experiment: trains the centroid-guided
contrastive autoencoder, applies latent-space pf-SMOTE oversampling, then
trains and evaluates the downstream ResNet classifier — the full pipeline
that produces every "DeepPolyCL-SMOTE" number in Table 1 of the paper, for
all three datasets (MNIST, FMNIST, CIFAR-10) and all 10 folds.

Usage:
    pip install -r requirements.txt
    # Place {dataset}_folds_data.pth and {dataset}_imbalanced_data.pth for
    # each dataset under ./prepared_data/ (see README.md for the expected
    # format) or point DPC_DATA_DIR at wherever you generated them.
    python main.py                       # runs MNIST by default
    DPC_DATASET=fmnist python main.py    # switch dataset without editing the file
    DPC_DATASET=cifar10 python main.py

To switch the SMOTE variant or dataset, edit the CONFIG block below, or use
the DPC_DATASET / DPC_DATA_DIR / DPC_RESULTS_DIR / DPC_OUTPUT_ROOT
environment variables — everything else in this file is just the k-fold loop
and CSV bookkeeping. See README.md for the list of valid SMOTE_VARIANT values.
"""

import os
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from utils import run_deeppoly_clsmote

# ==============================================================================
# CONFIG — edit this block to change what gets run
# ==============================================================================

# Override with e.g. `DPC_DATASET=fmnist nohup python run_deeppoly_clsmote.py ...`
# so the same file can be launched on multiple nodes at once (one dataset each)
# without editing it between launches.
DATASET = os.environ.get('DPC_DATASET', 'mnist')  # 'mnist' | 'fmnist' | 'cifar10'

# Any class name from the `smote_variants` package, e.g.:
#   'polynom_fit_SMOTE_bus', 'polynom_fit_SMOTE_star',
#   'polynom_fit_SMOTE_mesh', 'polynom_fit_SMOTE_poly', 'SMOTE'
# 'mesh' is the paper's stated default topology (Implementation Details).
SMOTE_VARIANT = 'polynom_fit_SMOTE_mesh'
SMOTE_KWARGS = {}  # extra kwargs forwarded to the sampler, e.g. {'topology': 'poly_2'}

# Exact per-dataset hyperparameters as reported in the paper's "Implementation
# Details" section (main_rv1.tex, ~line 400). Previously this script (and the
# original main_DeepPolyDLSMOTE.py it was extracted from) hardcoded ONE set of
# values — alpha=0.9, temperature=0.07, n_z=600, weight_decay=1e-4 — which is
# actually the CIFAR-10 config, and used it for every dataset regardless of
# which one was selected. Reviewer 2 (Validity) flagged exactly this class of
# risk: "manuscript, code, and reported validation setting are not sufficiently
# aligned." Do not silently change these without also updating the paper.
PAPER_HYPERPARAMS = {
    'mnist':   {'lr': 2e-4, 'alpha': 0.7, 'temperature': 0.1,  'n_z': 300, 'weight_decay': 5e-4, 'epochs': 200},
    'fmnist':  {'lr': 2e-4, 'alpha': 0.5, 'temperature': 0.07, 'n_z': 300, 'weight_decay': 5e-4, 'epochs': 200},
    'cifar10': {'lr': 2e-4, 'alpha': 0.9, 'temperature': 0.07, 'n_z': 600, 'weight_decay': 1e-4, 'epochs': 250},
}

# Set to a list (e.g. [0.3, 0.5, 0.7]) to sweep alpha instead of using the
# paper's fixed value for DATASET — leave as None for the authoritative,
# paper-matching run whose results are meant to go back into the manuscript.
ALPHA_GRID = None

HYPERPARAMS = dict(PAPER_HYPERPARAMS[DATASET])
HYPERPARAMS['warmup_epochs'] = 0

if ALPHA_GRID is None:
    ALPHA_GRID = [HYPERPARAMS['alpha']]

MODEL_ARGS = {
    'dim_h': 64,
    'n_channel': 3 if DATASET == 'cifar10' else 1,
    'num_class': 10,
    'batch_size': 256,
    # 28x28 (MNIST/FMNIST) bottlenecks to a 3x3 spatial map after the conv
    # stack; CIFAR-10, at its native 32x32, bottlenecks to 4x4 instead.
    'bottleneck_size': 4 if DATASET == 'cifar10' else 3,
}

# CIFAR-10 must be trained at its native 32x32 resolution (as stated in the
# paper's Datasets subsection), not a 28x28 file — so its prepared-data
# filenames carry a '32' suffix; MNIST/FMNIST have no suffix.
DATA_SUFFIX = '32' if DATASET == 'cifar10' else ''

# Set to an int (e.g. 3) to only run the first N folds for a quick check.
# Set to None (the default for an authoritative run) to use all 10 folds.
NUM_FOLDS = None

DATA_DIR = Path(os.environ.get('DPC_DATA_DIR', './prepared_data'))
RESULTS_DIR = Path(os.environ.get('DPC_RESULTS_DIR', './results'))
OUTPUT_ROOT = os.environ.get('DPC_OUTPUT_ROOT', './runs')

# ==============================================================================


def main():
    time_start = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # A partial-folds probe writes to its own file (suffix'd with the fold
    # count) so its "Average" rows — based on fewer folds — never mix with,
    # or overwrite, the full-run results file for this dataset+variant.
    suffix = f"_probe{NUM_FOLDS}folds" if NUM_FOLDS is not None else ""
    results_filepath = RESULTS_DIR / f"Results_DeepPolyCLSMOTE_{DATASET}_{SMOTE_VARIANT}{suffix}.csv"

    # Keep any previously saved rows (e.g. the alpha=0.9 run already on disk)
    # so different alpha values stay comparable in one file instead of one
    # sweep overwriting the last.
    if results_filepath.exists():
        existing = pd.read_csv(results_filepath)
        # Drop stale rows for any alpha this run is about to (re)generate —
        # otherwise a killed/rerun attempt leaves duplicate Fold rows (e.g. an
        # old, incomplete fold 1 alongside the fresh one) that silently
        # contaminate the Average row computed below.
        stale_mask = existing['alpha'].isin(ALPHA_GRID)
        if stale_mask.any():
            print(f"Dropping {stale_mask.sum()} stale row(s) for alpha(s) {ALPHA_GRID} "
                  f"from a previous incomplete/older run.")
        results_list = existing[~stale_mask].to_dict('records')
        print(f"Loaded {len(results_list)} existing rows from {results_filepath}")
    else:
        results_list = []

    print(f"Loading {DATASET} data from {DATA_DIR} ...")
    folds_data = torch.load(DATA_DIR / f"{DATASET}_folds_data{DATA_SUFFIX}.pth", weights_only=False)
    imbalanced_train_dataset_list = torch.load(DATA_DIR / f"{DATASET}_imbalanced_data{DATA_SUFFIX}.pth", weights_only=False)
    if NUM_FOLDS is not None:
        folds_data = folds_data[:NUM_FOLDS]
        imbalanced_train_dataset_list = imbalanced_train_dataset_list[:NUM_FOLDS]
    print(f"Loaded {len(folds_data)} folds.")

    for alpha in ALPHA_GRID:
        params = dict(HYPERPARAMS)
        params['alpha'] = alpha
        params['dataset_name'] = DATASET

        fold_f1, fold_asca, fold_gm, fold_perclass = [], [], [], []

        for fold_idx, (fold_loaders, fold_imbalanced) in enumerate(zip(folds_data, imbalanced_train_dataset_list)):
            print(f"\n--- alpha={alpha} | Fold {fold_idx + 1}/{len(folds_data)} | "
                  f"dataset={DATASET} | smote={SMOTE_VARIANT} ---")

            f1, asca, g_mean, per_class_results = run_deeppoly_clsmote(
                params=params,
                fold_data_loaders=fold_loaders,
                imbalanced_train_list=fold_imbalanced,
                model_args=MODEL_ARGS,
                smote_variant=SMOTE_VARIANT,
                smote_kwargs=SMOTE_KWARGS,
                output_root=OUTPUT_ROOT,
            )

            fold_f1.append(f1)
            fold_asca.append(asca)
            fold_gm.append(g_mean)
            fold_perclass.append(per_class_results)

            row = dict(params)
            row.update({
                'smote_variant': SMOTE_VARIANT,
                'Fold': fold_idx + 1,
                'F1': f1, 'ASCA': asca, 'GM': g_mean,
                'Perclass_Report': per_class_results,
            })
            results_list.append(row)

            pd.DataFrame(results_list).to_csv(results_filepath, index=False)
            print(f"... saved alpha={alpha} fold {fold_idx + 1} results to {results_filepath}")

        avg_row = dict(params)
        avg_row.update({
            'smote_variant': SMOTE_VARIANT,
            'Fold': 'Average',
            'F1': np.mean(fold_f1), 'ASCA': np.mean(fold_asca), 'GM': np.mean(fold_gm),
            'Std_F1': np.std(fold_f1), 'Std_ASCA': np.std(fold_asca), 'Std_GM': np.std(fold_gm),
            'Perclass_Report': str(fold_perclass),
        })
        results_list.append(avg_row)
        pd.DataFrame(results_list).to_csv(results_filepath, index=False)

        print(f"\nalpha={alpha} done. Avg F1={np.mean(fold_f1):.4f} (+/-{np.std(fold_f1):.4f})  "
              f"Avg ASCA={np.mean(fold_asca):.4f} (+/-{np.std(fold_asca):.4f})")

    print(f"\nAll alphas done. Results saved to: {results_filepath}")
    print(f"Time spent (hr): {(time.time() - time_start) / 3600:.2f}")


if __name__ == "__main__":
    main()

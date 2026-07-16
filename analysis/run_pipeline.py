#!/usr/bin/env python3
"""
Full analysis pipeline runner.

Runs all analysis scripts in order:
  Compute (metric-independent):
    10  → RDMs for MLP/RL tasks
    10b → RDMs for adding (phase-based)
    10c → RDMs for mnist_rnn (temporal)
    16  → Dimensionality (participation ratio)

  Per metric (cosine and/or pearson):
    11  → RSA validity / noise ceiling
    11b → Temporal validity (adding phases, mnist_rnn)
    13  → RDM gallery
    14  → Category structure
    20  → HP effects
    21  → Latent variable analysis
    22  → PCA on RDMs

Usage:
    python run_pipeline.py                     # cosine only
    python run_pipeline.py --metric pearson
    python run_pipeline.py --metric both
    python run_pipeline.py --skip-compute      # skip scripts 10/10b/10c/16
"""

import argparse
import subprocess
import sys
from pathlib import Path

ANALYSIS = Path(__file__).parent
PYTHON = sys.executable

COMPUTE_SCRIPTS = [
    "10_compute_rdms.py",
    "10b_compute_adding_phases.py",
    "10c_compute_temporal_rdm_rnn.py",
    "16_dimensionality.py",
]

METRIC_SCRIPTS = [
    "11_rsa_validity.py",
    "11b_temporal_validity.py",
    "13_rdm_gallery.py",
    "14_category_structure.py",
    "20_hp_effects.py",
    "21_latent_vars.py",
    "22_rdm_pca.py",
]


def run(script, extra_args=()):
    cmd = [PYTHON, str(ANALYSIS / script)] + list(extra_args)
    print(f"\n{'='*60}", flush=True)
    print(f"Running: {' '.join(cmd)}", flush=True)
    print('='*60, flush=True)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\nERROR: {script} exited with code {result.returncode}", file=sys.stderr)
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(description="Run the full analysis pipeline.")
    parser.add_argument("--metric", choices=["cosine", "pearson", "both"], default="cosine",
                        help="RDM metric(s) to use for downstream scripts (default: cosine).")
    parser.add_argument("--skip-compute", action="store_true",
                        help="Skip scripts 10/10b/10c/16 (RDM computation and dimensionality).")
    args = parser.parse_args()

    metrics = ["cosine", "pearson"] if args.metric == "both" else [args.metric]

    if not args.skip_compute:
        for script in COMPUTE_SCRIPTS:
            run(script)

    for metric in metrics:
        print(f"\n{'#'*60}", flush=True)
        print(f"# Metric: {metric}", flush=True)
        print(f"{'#'*60}", flush=True)
        for script in METRIC_SCRIPTS:
            run(script, ["--metric", metric])


if __name__ == "__main__":
    main()

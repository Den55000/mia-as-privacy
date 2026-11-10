# [AI-generated: Claude Code, Claude Opus 5.5, 2026-11-10] Entire file.
"""
Compare attack predictions of several runs on the same data split, e.g. to check that the
LiRA loader-worker change leaves predictions and metrics consistent (supervisors' condition,
2026-11-10), or to compare target models trained at batch size 512 and 128.

For every attack it prints, per run: AUC, TPR at 1% FPR, and the Jaccard index between the
members it detects at 1% FPR and those the FIRST run detects. Read the first run's row
against the spread across the other runs: a change is consistent if its numbers stay
within the seed-to-seed variation.

Usage (from experiment/mia_comp/):
    python experiment_scripts/compare_preds.py <attack_set_membership.npy> <label>=<preds_dir> ...
  where <preds_dir> = <DATA_DIR>/<EXP_NAME>/preds_sd<seed>/<dataset>/<arch>
"""
import os
import sys

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

ATTACKS = ["lira", "reference", "losstraj", "calibration", "aug", "shokri", "yeom"]
FPR = 0.01


def tpr_at_fpr(labels, scores, fpr_target):
    fpr, tpr, _ = roc_curve(labels, scores)
    return tpr[np.searchsorted(fpr, fpr_target, side="right") - 1]


def detected(labels, scores, fpr_target):
    """Members scoring above the threshold that flags fpr_target of the non-members."""
    thr = np.quantile(scores[labels == 0], 1 - fpr_target)
    return set(np.where((scores > thr) & (labels == 1))[0])


def main():
    labels = np.load(sys.argv[1])
    runs = [a.split("=", 1) for a in sys.argv[2:]]
    print(f"{'attack':12} {'run':22} {'AUC':>6} {'TPR@1%':>7} {'Jaccard vs ' + runs[0][0]:>24}")
    for attack in ATTACKS:
        first = None
        for label, d in runs:
            path = os.path.join(d, attack, f"pred_{attack}.npy")
            if not os.path.exists(path):
                print(f"{attack:12} {label:22} {'missing':>6}")
                continue
            scores = np.load(path)
            det = detected(labels, scores, FPR)
            if first is None:
                first = det
            jac = f"{len(det & first) / len(det | first):.2f}" if det | first else "n/a (none detected)"
            print(f"{attack:12} {label:22} {roc_auc_score(labels, scores):6.3f} "
                  f"{tpr_at_fpr(labels, scores, FPR):7.3f} {jac:>24}")
        print()


if __name__ == "__main__":
    main()

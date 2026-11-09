# [AI-generated: Claude Code, Claude Opus 5.5, 2026-10-23] Entire file.
"""
Sanity-check the outputs of prepare_target.sh and obtain_pred.sh.

For every target model it prints the last logged train/test accuracy. For every
pred_<attack>.npy it checks the shape and prints AUC, accuracy at threshold 0.5 and
TPR at 1% / 0.1% FPR against the ground-truth membership labels.

Usage (from experiment/mia_comp/):
    python experiment_scripts/check_preds.py <DATA_DIR>/<EXP_NAME> [seed]
"""
import glob
import os
import sys

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def tpr_at_fpr(labels, scores, fpr_target):
    fpr, tpr, _ = roc_curve(labels, scores)
    return tpr[np.searchsorted(fpr, fpr_target, side="right") - 1]


def main():
    exp_dir = sys.argv[1]
    seed = sys.argv[2] if len(sys.argv) > 2 else "0"
    ok = True

    print("== Target models ==")
    logs = sorted(glob.glob(os.path.join(exp_dir, "target", "target_models", "*", "*", "*.log")))
    if not logs:
        print("  no target model logs found")
        ok = False
    for log in logs:
        lines = [l for l in open(log) if "train_acc" in l]
        last = lines[-1].split(" - ", 1)[-1].strip() if lines else "no epochs logged"
        print(f"  {os.path.relpath(log, exp_dir)}: {last}")

    print(f"\n== Attack predictions, seed {seed} ==")
    preds_dir = os.path.join(exp_dir, f"preds_sd{seed}")
    files = sorted(glob.glob(os.path.join(preds_dir, "*", "*", "*", "pred_*.npy")))
    if not files:
        print(f"  no predictions found under {preds_dir}")
        ok = False
    print(f"  {'dataset/arch/attack':40s} {'shape':>9s} {'AUC':>6s} {'acc@0.5':>8s} {'TPR@1%':>7s} {'TPR@0.1%':>8s}")
    for f in files:
        dataset, arch, attack = os.path.relpath(f, preds_dir).split(os.sep)[:3]
        membership = np.load(os.path.join(exp_dir, "target", dataset, "attack_set_membership.npy"))
        pred = np.load(f)
        name = f"{dataset}/{arch}/{attack}"
        if pred.shape != membership.shape or not np.all(np.isfinite(pred)):
            print(f"  {name:40s} BAD: shape {pred.shape} (expected {membership.shape}), "
                  f"{np.sum(~np.isfinite(pred))} non-finite values")
            ok = False
            continue
        auc = roc_auc_score(membership, pred)
        acc = np.mean((pred > 0.5) == membership)
        print(f"  {name:40s} {str(pred.shape):>9s} {auc:6.3f} {acc:8.3f} "
              f"{tpr_at_fpr(membership, pred, 0.01):7.3f} {tpr_at_fpr(membership, pred, 0.001):8.3f}")

    print("\nAll checks passed." if ok else "\nSome checks FAILED.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

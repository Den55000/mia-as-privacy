# [AI-generated: Claude Code, Claude Opus 5.5, 2026-11-10] Entire file.
"""
Time LiRA shadow-model training with a given number of data-loader workers (pilot report,
"Why the GPUs sat idle"). Builds the same training set as LiraAttack.infer (auxiliary set +
attack set) and calls LIRAUtil.train_shadow_models with the paper's settings, but only
for a few shadow models, into a scratch folder.

Usage (from experiment/mia_comp/):
    python experiment_scripts/bench_shadow.py <DATA_DIR>/<EXP_NAME> <dataset> <arch> <epochs> \
        <num_shadow_models> <num_workers> <scratch_dir>
"""
import os
import pickle
import sys
import time

import torch
from torch.utils.data import ConcatDataset

sys.path.append(os.path.join(os.getcwd(), "..", ".."))
from experiment import models  # noqa: E402
from miae.attacks.lira_mia import LiraAuxiliaryInfo, LIRAUtil  # noqa: E402

NUM_CLASSES = {"cifar10": 10, "cifar100": 100}


def main():
    exp_dir, dataset, arch, epochs, num_shadow, workers, scratch = sys.argv[1:8]
    data_dir = os.path.join(exp_dir, "target", dataset)
    sets = {}
    for name in ["aux_set", "target_trainset", "target_testset"]:
        with open(os.path.join(data_dir, f"{name}.pkl"), "rb") as f:
            sets[name] = pickle.load(f)
    # as in LiraAttack.infer: shadow models train on halves of (auxiliary set + attack set)
    full = ConcatDataset([sets["aux_set"], ConcatDataset([sets["target_trainset"], sets["target_testset"]])])

    # as obtain_pred.get_aux_info passes for LiRA, plus num_workers
    info = LiraAuxiliaryInfo({'device': 'cuda', 'seed': 0, 'save_path': scratch, 'num_classes': NUM_CLASSES[dataset],
                              'lr': 0.1, 'num_shadow_models': int(num_shadow), 'epochs': int(epochs),
                              'shadow_path': os.path.join(scratch, f"shadow_w{workers}"), 'shadow_diff_init': True,
                              'num_workers': int(workers)})
    model = models.get_model(arch, NUM_CLASSES[dataset], 32)

    torch.cuda.synchronize()
    start = time.time()
    LIRAUtil.train_shadow_models(model, full, info)
    torch.cuda.synchronize()
    per_shadow = (time.time() - start) / int(num_shadow) / 60
    print(f"RESULT {dataset} {arch} workers={workers}: {num_shadow} shadows, {per_shadow:.1f} min per shadow "
          f"-> 20 shadows = {20 * per_shadow / 60:.1f} h")


if __name__ == "__main__":
    main()

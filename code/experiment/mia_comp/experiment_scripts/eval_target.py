# [AI-generated: Claude Code, Claude Opus 5.5, 2026-11-10] Entire file.
"""
Re-measure a trained target model's accuracy, to tell a badly trained model from
misleading logging (see the pilot report, "Problems found").

train_target_model() in obtain_pred.py logs train/test accuracy on *augmented* images
(random crop + flip), in eval mode. This script prints, for the same saved model:
  train, augmented,   eval mode   -> should reproduce the logged train_acc
  train, clean,       eval mode   -> the model's real train accuracy
  train, clean,       train mode  -> BatchNorm with batch statistics instead of running ones
  test,  clean,       eval mode   -> the model's real test accuracy
High clean/eval numbers: the model is fine and only the log is misleading.
Low eval but high train-mode numbers: a BatchNorm running-statistics problem.
Low everywhere: the model is undertrained (e.g. too few updates at batch size 512).

Usage (from experiment/mia_comp/):
    python experiment_scripts/eval_target.py <DATA_DIR>/<EXP_NAME> <dataset> <arch>
"""
import copy
import os
import pickle
import sys

import torch
import torchvision.transforms as T
from torch.utils.data import ConcatDataset, DataLoader, Subset

sys.path.append(os.path.join(os.getcwd(), "..", ".."))
from experiment import models  # noqa: E402

NUM_CLASSES = {"cifar10": 10, "cifar100": 100}
# same normalisation as datasets.get_cifar10 / get_cifar100, without the augmentation
CLEAN = T.Compose([T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


def set_transform(ds, transform):
    """Replace the transform of every torchvision dataset under nested Subset/ConcatDataset wrappers."""
    if isinstance(ds, Subset):
        set_transform(ds.dataset, transform)
    elif isinstance(ds, ConcatDataset):
        for d in ds.datasets:
            set_transform(d, transform)
    else:
        ds.transform = transform


@torch.no_grad()
def accuracy(model, ds, device, train_mode=False):
    model.train(train_mode)
    correct = 0
    for x, y in DataLoader(ds, batch_size=512, shuffle=False, num_workers=8):
        correct += (model(x.to(device)).argmax(1) == y.to(device)).sum().item()
    return correct / len(ds)


def main():
    exp_dir, dataset, arch = sys.argv[1:4]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir = os.path.join(exp_dir, "target", dataset)
    with open(os.path.join(data_dir, "target_trainset.pkl"), "rb") as f:
        trainset = pickle.load(f)
    with open(os.path.join(data_dir, "target_testset.pkl"), "rb") as f:
        testset = pickle.load(f)

    model_file = os.path.join(exp_dir, "target", "target_models", dataset, arch, f"target_model_{arch}{dataset}.pkl")
    model = models.get_model(arch, NUM_CLASSES[dataset], 32).to(device)
    model.load_state_dict(torch.load(model_file, map_location=device))

    print(f"== {model_file}")
    print(f"  train, augmented, eval mode : {accuracy(model, trainset, device):.4f}   (what the training log reports)")
    set_transform(trainset, CLEAN)
    set_transform(testset, CLEAN)
    print(f"  train, clean,     eval mode : {accuracy(model, trainset, device):.4f}")
    # a copy, because train mode updates the BatchNorm running statistics
    print(f"  train, clean,     train mode: {accuracy(copy.deepcopy(model), trainset, device, train_mode=True):.4f}")
    print(f"  test,  clean,     eval mode : {accuracy(model, testset, device):.4f}")


if __name__ == "__main__":
    main()

#!/bin/bash
# [AI-generated: Claude Code, Claude Opus 5.5, 2026-10-23] Changes to the authors' script: settings via environment variables, stop at first error, per-run scratch folders, sequential seeds. See docs/REPRO_LOG.md.
# Partition the dataset into target and auxiliary sets, then train the target model(s).
#
# Run from experiment/mia_comp/. Every setting can be overridden with an environment
# variable; the defaults are the paper's settings. Example:
#   DATASETS="cifar10" ARCHS="resnet56" bash experiment_scripts/prepare_target.sh
#
#   DATA_DIR      where results go (default: <project root>/data)
#   DATASETS      space-separated list (default: "cifar10 cifar100 cinic10")
#   ARCHS         space-separated list (default: "resnet56 wrn32_4 vgg16 mobilenet")
#   SHUFFLE_SEED  1 for the main run; 2-4 for the repeated partitions (default: 1)
#   EXP_NAME      experiment folder under DATA_DIR (default: miae_standard_exp)
#   EPOCHS        override the per-dataset epoch count (smoke tests only)
set -euo pipefail

# PyTorch >= 2.6 refuses torch.load of whole objects by default; the attack code needs it.
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

DATA_DIR="${DATA_DIR:-$(realpath ../../../data)}"
read -r -a datasets <<< "${DATASETS:-cifar10 cifar100 cinic10}"
read -r -a archs <<< "${ARCHS:-resnet56 wrn32_4 vgg16 mobilenet}"
# for tabular data: DATASETS="purchase100 texas100" ARCHS="mlp_for_texas_purchase"

seed=0 # keep seed = 0
# for repeat training, we do shuffle_seed from 2 to 4 to reshuffle the target-auxiliary dataset partition
# (e.g. EXP_NAME=repeat_miae_standard_exp/miae_standard_exp_2 SHUFFLE_SEED=2)
shuffle_seed="${SHUFFLE_SEED:-1}"
exp_name="${EXP_NAME:-miae_standard_exp}"
data_dir="${DATA_DIR}/${exp_name}/target"
mkdir -p "$data_dir"

target_model_path="$data_dir/target_models"

echo "DATA_DIR=$DATA_DIR datasets=(${datasets[*]}) archs=(${archs[*]}) shuffle_seed=$shuffle_seed"

for dataset in "${datasets[@]}"; do
  # if assign different num_epoch for different dataset
  if [ "$dataset" == "cifar10" ]; then
    num_epoch=60
  elif [ "$dataset" == "cifar100" ]; then
    num_epoch=100
  elif [ "$dataset" == "cinic10" ]; then
    num_epoch=60
  elif [ "$dataset" == "purchase100" ]; then
    num_epoch=30
  elif [ "$dataset" == "texas100" ]; then
    num_epoch=30
  else
    echo "Error: Unknown dataset $dataset"
    exit 1
  fi
  num_epoch="${EPOCHS:-$num_epoch}"

  mkdir -p "$data_dir/$dataset"
  # save the dataset (skip if already saved, so the partition is never silently re-made)
  if [ -f "$data_dir/$dataset/attack_set_membership.npy" ]; then
    echo "Dataset $dataset already saved, skip"
  else
    echo "Saving dataset $dataset"
    # --dataset_file_root is needed for CINIC-10 (fix #10)
    python3 obtain_pred.py --dataset "$dataset" --save_dataset "True" --data_path "$data_dir" --seed "$seed" \
      --data_aug "True" --shuffle_seed "$shuffle_seed" --dataset_file_root "$data_dir"
  fi
    for arch in "${archs[@]}"; do
      # for each arch, train the target model
      mkdir -p "$target_model_path/$dataset/$arch"
      target_model_save_path="$target_model_path/$dataset/$arch"
      echo "Obtaining target_model for $dataset $arch"
      # if the target model is already trained, then skip
      if [ -f "$target_model_save_path/target_model_$arch$dataset.pkl" ]; then
        echo "Target model for $dataset $arch already exists, skip"
        continue
      fi
      python3 obtain_pred.py --train_target_model "True" --dataset "$dataset" --target_model "$arch" \
       --seed "$seed" --delete-files "True" --data_aug "True"  --target_model_path "$target_model_save_path" \
       --attack_epochs "$num_epoch" --target_epochs "$num_epoch" --data_path "$data_dir" --shuffle_seed "$shuffle_seed"
    done
done

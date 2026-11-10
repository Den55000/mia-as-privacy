#!/bin/bash
# [AI-generated: Claude Code, Claude Opus 5.5, 2026-10-23] Changes to the authors' script: settings via environment variables, stop at first error, per-run scratch folders, sequential seeds. See docs/REPRO_LOG.md.
# Obtain the predictions of the attacks on the target models, for one seed.
#
# Run from experiment/mia_comp/:  bash experiment_scripts/obtain_pred.sh <seed>
# Every setting can be overridden with an environment variable; the defaults are the
# paper's settings. Example:
#   DATASETS="cifar10" ARCHS="resnet56" MIAS="yeom calibration" bash experiment_scripts/obtain_pred.sh 0
#
#   DATA_DIR   where results go (default: <project root>/data)
#   DATASETS   space-separated list (default: "cifar10 cifar100 cinic10")
#   ARCHS      space-separated list (default: "resnet56 wrn32_4 vgg16 mobilenet")
#   MIAS       space-separated list; keep lira before reference (default: all 7)
#   EXP_NAME   experiment folder under DATA_DIR (default: miae_standard_exp)
#   PREP_ROOT  scratch space for temporary attack files (default: the preds folder;
#              on SLURM use $TMPDIR)
#   DEVICE     torch device (default: cuda:0)
#   SHADOW_WORKERS  data-loader workers for LiRA shadow-model training (default: 0, the
#              authors' setting; 8 is ~2.2x faster, see REPRO_LOG.md 2026-11-10)
#   EPOCHS     override the per-dataset epoch count (smoke tests only)
set -euo pipefail

# PyTorch >= 2.6 refuses torch.load of whole objects by default; the attack code needs it.
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

DATA_DIR="${DATA_DIR:-$(realpath ../../../data)}"
read -r -a datasets <<< "${DATASETS:-cifar10 cifar100 cinic10}"
read -r -a archs <<< "${ARCHS:-resnet56 wrn32_4 vgg16 mobilenet}"
read -r -a mias <<< "${MIAS:-lira reference shokri losstraj calibration yeom aug}"
# for tabular data: DATASETS="purchase100 texas100" ARCHS="mlp_for_texas_purchase" (no "aug")

seed="${1:-0}"
echo "obtain_pred.sh seed = $seed"

exp_name="${EXP_NAME:-miae_standard_exp}"
data_dir="${DATA_DIR}/${exp_name}/target"
preds_dir="${DATA_DIR}/${exp_name}/preds_sd${seed}"
target_model_path="$data_dir/target_models"
device="${DEVICE:-cuda:0}"

mkdir -p "$preds_dir"
echo "DATA_DIR=$DATA_DIR datasets=(${datasets[*]}) archs=(${archs[*]}) mias=(${mias[*]})"

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

    for arch in "${archs[@]}"; do
      # for a given dataset and architecture, save the predictions
      mkdir -p "$preds_dir/$dataset/$arch"

      # prepare a directory for lira shadow models so lira and other attacks (RMIA, reference) could share the same shadow models
      lira_shadow_dir="$preds_dir/$dataset/$arch/lira_shadow_ckpts"
      mkdir -p "$lira_shadow_dir"

        for mia in "${mias[@]}"; do
            result_dir="$preds_dir/$dataset/$arch/${mia}"
            # if the predictions are already saved, skip
            if [ -f "$result_dir/pred_$mia.npy" ]; then
                echo "Predictions already saved for $dataset $arch $mia at $result_dir/pred_$mia.npy"
                continue
            else
                echo "Predictions not saved for $dataset $arch $mia at $result_dir/pred_$mia.npy"
            fi

            # One scratch folder per (seed, dataset, arch, attack), so parallel jobs never
            # delete each other's files (fix #11). The original used one folder per seed.
            prepare_dir="${PREP_ROOT:-$preds_dir}/prepare_sd${seed}_${dataset}_${arch}_${mia}"
            if [ -d "$prepare_dir" ] ; then
                rm -r "$prepare_dir"
            fi

            mkdir -p "$result_dir"

            echo "Running $dataset $arch $mia ($(date))"
            target_model_save_path="$target_model_path/$dataset/$arch"

            python3 obtain_pred.py \
            --dataset "$dataset"\
            --target_model "$arch"\
            --attack "$mia"\
            --result_path "$result_dir"\
            --seed "$seed"\
            --delete-files "True" \
            --preparation_path "$prepare_dir" \
            --data_aug "False"  \
            --target_model_path "$target_model_save_path" \
            --attack_epochs "$num_epoch" \
            --target_epochs "$num_epoch" \
            --data_path "$data_dir" \
            --device "$device" \
            --dataset_file_root="$data_dir" \
            --lira_shadow_path "$lira_shadow_dir" \
            --shadow_num_workers "${SHADOW_WORKERS:-0}"

            rm -r "$prepare_dir"
            echo "Finished $dataset $arch $mia ($(date))"
        done
    done
done

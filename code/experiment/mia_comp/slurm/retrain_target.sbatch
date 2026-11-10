#!/bin/bash
# [AI-generated: Claude Code, Claude Opus 5.5, 2026-11-10] Entire file.
# Retrains the CIFAR-100/WRN-32-4 target model with the shadow models' batch size (pilot
# report, "Problems found"). obtain_pred.py trains the target at --batch_size 512, while
# LiRA's shadow models train at 128 (LiraAuxiliaryInfo reads 'shadow_batchsize', default 128).
# The data split is copied from pilot_wrn, so only the batch size differs.
# Run from experiment/mia_comp/:  sbatch slurm/retrain_target.sbatch
#   BATCH_SIZE (default 128) and EXP_NAME (default pilot_wrn_bs$BATCH_SIZE) can be overridden.
#SBATCH --job-name=mia_retarget
#SBATCH --partition=a100
#SBATCH --qos=a100-6hours
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=16G
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
source ../../.venv/bin/activate
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
DATA_DIR="${DATA_DIR:-$HOME/mia_data_NOBACKUP}"
BATCH_SIZE="${BATCH_SIZE:-128}"
out="$DATA_DIR/${EXP_NAME:-pilot_wrn_bs$BATCH_SIZE}"
data_dir="$out/target"
model_dir="$data_dir/target_models/cifar100/wrn32_4"
mkdir -p "$data_dir/cifar100" "$model_dir"
cp -n "$DATA_DIR"/pilot_wrn/target/cifar100/*.pkl "$DATA_DIR"/pilot_wrn/target/cifar100/*.npy "$data_dir/cifar100/"

echo "job $SLURM_JOB_ID batch_size=$BATCH_SIZE out=$out node=$(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader)"
start=$SECONDS
python3 obtain_pred.py --train_target_model "True" --dataset cifar100 --target_model wrn32_4 \
  --seed 0 --target_model_path "$model_dir" --target_epochs 100 --attack_epochs 100 \
  --batch_size "$BATCH_SIZE" --data_path "$data_dir" --shuffle_seed 1
echo "$(date '+%F %T') $SLURM_JOB_ID target_cifar100_wrn32_4_bs$BATCH_SIZE $(( SECONDS - start )) s" | tee -a "$out/timings.log"

cat "$model_dir"/*.log
python experiment_scripts/eval_target.py "$out" cifar100 wrn32_4

#!/bin/bash
# [AI-generated: Claude Code, Claude Opus 5.5, 2026-11-10] Entire file.
# Re-measures the pilot's target models (pilot report, "Problems found"): the suspect
# CIFAR-100/WRN-32-4 model, and the CIFAR-10/ResNet-56 model as a control.
# Run from experiment/mia_comp/:  sbatch slurm/eval_target.sbatch
#SBATCH --job-name=mia_evaltarget
#SBATCH --partition=a100
#SBATCH --qos=a100-30min
#SBATCH --time=00:20:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=16G
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
source ../../.venv/bin/activate
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
DATA_DIR="${DATA_DIR:-$HOME/mia_data_NOBACKUP}"

python experiment_scripts/eval_target.py "$DATA_DIR/pilot_wrn" cifar100 wrn32_4
python experiment_scripts/eval_target.py "$DATA_DIR/pilot_l40s" cifar10 resnet56

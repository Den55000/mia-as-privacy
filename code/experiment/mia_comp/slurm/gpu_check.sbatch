#!/bin/bash
# [AI-generated: Claude Code, Claude Opus 5.5, 2026-11-09] Entire file.
# 5-minute check that the venv sees a GPU on both pilot partitions, before queueing long jobs.
# Submit from experiment/mia_comp/:
#   sbatch slurm/gpu_check.sbatch                                          # A100
#   sbatch --partition=l40s --qos=l40s-30min slurm/gpu_check.sbatch       # L40S
#SBATCH --job-name=mia_gpucheck
#SBATCH --partition=a100
#SBATCH --qos=a100-30min
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
source ../../.venv/bin/activate
hostname; nvidia-smi
python - <<'EOF'
import torch
assert torch.cuda.is_available(), "no GPU visible"
print(torch.__version__, torch.cuda.get_device_name(0))
x = torch.randn(4096, 4096, device="cuda"); print("matmul OK", (x @ x).sum().item())
EOF

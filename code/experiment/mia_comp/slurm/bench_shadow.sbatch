#!/bin/bash
# [AI-generated: Claude Code, Claude Opus 5.5, 2026-11-10] Entire file.
# Measures the speed-up from data-loader workers in LiRA shadow-model training (pilot
# report, "Why the GPUs sat idle"): 2 CIFAR-10/ResNet-56 shadow models with 0 workers
# (the authors' code), then 2 with 8, on the same node. Reuses the pilot_l40s data split.
# Run from experiment/mia_comp/, once per partition:
#   sbatch -p a100 --qos=a100-6hours slurm/bench_shadow.sbatch
#   sbatch -p l40s --qos=l40s-6hours slurm/bench_shadow.sbatch
#SBATCH --job-name=mia_benchshadow
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=16G
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
source ../../.venv/bin/activate
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
DATA_DIR="${DATA_DIR:-$HOME/mia_data_NOBACKUP}"
WORKERS="${WORKERS:-0 8}"

echo "job $SLURM_JOB_ID partition=$SLURM_JOB_PARTITION node=$(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "CPU: $(lscpu | grep 'Model name' | sed 's/.*: *//')"
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv -l 30 > "logs/gpu_bench_${SLURM_JOB_ID}.csv" &

for w in $WORKERS; do
  python experiment_scripts/bench_shadow.py "$DATA_DIR/pilot_l40s" cifar10 resnet56 60 2 "$w" "$TMPDIR/bench"
done

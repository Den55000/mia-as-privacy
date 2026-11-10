#!/bin/bash
# [AI-generated: Claude Code, Claude Opus 5.5, 2026-11-10] Entire file.
# One seed set per array task: all 7 attacks for one (experiment, dataset, arch, seed), one after
# another (LiRA first, so Reference can reuse its shadow models). Submitted by slurm/submit_stage1.sh;
# each line of the task file $TASKS is "<EXP_NAME> <dataset> <arch> <seed>".
# LiRA shadow models train with $SHADOW_WORKERS data-loader workers (default 8; approved
# 2026-11-10, REPRO_LOG.md). A resubmitted task skips attacks whose predictions exist.
#SBATCH --job-name=mia_seedset
#SBATCH --partition=l40s
#SBATCH --qos=l40s-1day
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=16G
#SBATCH --output=logs/%x_%A_%a.out

set -euo pipefail
source ../../.venv/bin/activate
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export PREP_ROOT="$TMPDIR"   # attack scratch files on node-local disk (fix #11)
export SHADOW_WORKERS="${SHADOW_WORKERS:-8}"
export DATA_DIR="${DATA_DIR:-$HOME/mia_data_NOBACKUP}"
read -r EXP_NAME DATASETS ARCHS seed < <(sed -n "${SLURM_ARRAY_TASK_ID}p" "$TASKS")
export EXP_NAME DATASETS ARCHS
out="$DATA_DIR/$EXP_NAME"
mkdir -p "$out"

echo "job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} exp=$EXP_NAME $DATASETS/$ARCHS seed=$seed workers=$SHADOW_WORKERS" \
     "node=$(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader)"
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv -l 60 \
  > "$out/gpu_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.csv" &

# fix #12: a run killed mid-LiRA leaves a shadow-model folder without shadow.pth; remove it, or
# LiRA may count it as trained and skip training
shadow_dir="$out/preds_sd$seed/$DATASETS/$ARCHS/lira_shadow_ckpts"
for d in "$shadow_dir"/*/; do
  if [ -d "$d" ] && [ ! -f "$d/shadow.pth" ]; then echo "removing incomplete shadow model $d"; rm -r "$d"; fi
done

for mia in lira reference shokri losstraj calibration yeom aug; do
  [ -f "$out/preds_sd$seed/$DATASETS/$ARCHS/$mia/pred_$mia.npy" ] && { echo "$mia done already, skip"; continue; }
  start=$SECONDS
  MIAS="$mia" bash experiment_scripts/obtain_pred.sh "$seed"
  echo "$(date '+%F %T') $SLURM_JOB_PARTITION ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} sd${seed}_${DATASETS}_${ARCHS}_attack_$mia $(( SECONDS - start )) s" \
    | tee -a "$out/timings.log"
done

# other tasks of the same experiment may still be running, so a "FAILED" here can be theirs
python experiment_scripts/check_preds.py "$out" "$seed" || true

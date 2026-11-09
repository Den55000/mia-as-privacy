#!/bin/bash
# [AI-generated: Claude Code, Claude Opus 5.5, 2026-11-09] Entire file.
# One pilot step (PLAN.md, D5). Partition, QoS, walltime and the step are set by slurm/submit_pilot.sh:
#   STEP=prepare    data split + target model         (DATASETS/ARCHS from the caller)
#   STEP=lira       LiRA: 20 shadow models + queries
#   STEP=reference  Reference attack, reusing LiRA's shadow models
#   STEP=fast       losstraj, calibration, shokri, aug, yeom, one after another
#   STEP=wrn_probe  CIFAR-100/WRN-32-4: target model, then LiRA until the walltime kills it
# Results: $DATA_DIR/$EXP_NAME; per-step times in timings.log, GPU use in gpu_<jobid>.csv.
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --tmp=20G
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
source ../../.venv/bin/activate
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export PREP_ROOT="$TMPDIR"   # attack scratch files on node-local disk (fix #11)
out="$DATA_DIR/$EXP_NAME"
mkdir -p "$out"

echo "job $SLURM_JOB_ID step=$STEP exp=$EXP_NAME node=$(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader)"
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv -l 60 > "$out/gpu_${SLURM_JOB_ID}.csv" &

timed() {  # timed <label> <command...>: run a step and append its wall-clock time to timings.log
  local label="$1"; shift
  local start=$SECONDS
  "$@"
  echo "$(date '+%F %T') $SLURM_JOB_PARTITION $SLURM_JOB_ID $label $(( SECONDS - start )) s" | tee -a "$out/timings.log"
}

case "$STEP" in
  prepare)   timed "prepare_${DATASETS}_${ARCHS}" bash experiment_scripts/prepare_target.sh ;;
  lira)      MIAS=lira timed attack_lira bash experiment_scripts/obtain_pred.sh 0 ;;
  reference) MIAS=reference timed attack_reference bash experiment_scripts/obtain_pred.sh 0 ;;
  fast)
    for mia in losstraj calibration shokri aug yeom; do
      MIAS="$mia" timed "attack_$mia" bash experiment_scripts/obtain_pred.sh 0
    done ;;
  wrn_probe)
    timed "prepare_${DATASETS}_${ARCHS}" bash experiment_scripts/prepare_target.sh
    MIAS=lira timed attack_lira bash experiment_scripts/obtain_pred.sh 0 ;;
  *) echo "unknown STEP=$STEP"; exit 1 ;;
esac

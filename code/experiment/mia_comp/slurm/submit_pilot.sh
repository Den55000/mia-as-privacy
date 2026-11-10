#!/bin/bash
# [AI-generated: Claude Code, Claude Opus 5.5, 2026-11-09] Entire file.
# Submits the whole pilot (PLAN.md, D5, ~30 GPU-h). Run from experiment/mia_comp/ on login12:
#   bash slurm/submit_pilot.sh
#
# 1. CIFAR-10/ResNet-56, seed 0, all 7 attacks, once on l40s and once on a100:
#      prepare -> { lira -> reference , fast }
# 2. CIFAR-100/WRN-32-4 probe on a100: target model + LiRA for 6 h (killed on purpose), to
#    extrapolate the worst-case per-shadow time.
# Job IDs go to slurm/pilot_jobs.txt; afterwards run: bash slurm/pilot_report.sh
set -euo pipefail

export DATA_DIR="${DATA_DIR:-$HOME/mia_data_NOBACKUP}"   # *_NOBACKUP: excluded from the daily backup
mkdir -p logs "$DATA_DIR"
jobs_file=slurm/pilot_jobs.txt
: > "$jobs_file"

submit() {  # submit <label> <sbatch args...>: prints the job ID and records it
  local label="$1"; shift
  local id
  # set -e does not apply inside $(...), so stop explicitly; callers using $(submit ...) then stop too
  id=$(sbatch --parsable "$@" slurm/pilot.sbatch) || { echo "sbatch failed for $label, stopping" >&2; exit 1; }
  echo "$label $id" >> "$jobs_file"
  echo "submitted $label: $id" >&2
  echo "$id"
}

for part in l40s a100; do
  export EXP_NAME="pilot_$part" DATASETS=cifar10 ARCHS=resnet56
  prep=$(STEP=prepare submit "$part.prepare" -J "mia_prep_$part" -p "$part" --qos="$part-6hours" -t 03:00:00 --export=ALL)
  lira=$(STEP=lira submit "$part.lira" -J "mia_lira_$part" -p "$part" --qos="$part-1day" -t 1-00:00:00 \
           --dependency="afterok:$prep" --export=ALL)
  STEP=reference submit "$part.reference" -J "mia_ref_$part" -p "$part" --qos="$part-6hours" -t 04:00:00 \
    --dependency="afterok:$lira" --export=ALL >/dev/null
  STEP=fast submit "$part.fast" -J "mia_fast_$part" -p "$part" --qos="$part-6hours" -t 06:00:00 \
    --dependency="afterok:$prep" --export=ALL >/dev/null
done

EXP_NAME=pilot_wrn DATASETS=cifar100 ARCHS=wrn32_4 STEP=wrn_probe \
  submit "a100.wrn_probe" -J mia_wrnprobe -p a100 --qos=a100-6hours -t 06:00:00 --export=ALL >/dev/null

echo; echo "Submitted. Watch with: squeue --me    Results: $DATA_DIR/pilot_*"

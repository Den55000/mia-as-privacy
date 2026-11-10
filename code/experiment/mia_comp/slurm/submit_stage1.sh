#!/bin/bash
# [AI-generated: Claude Code, Claude Opus 5.5, 2026-11-10] Entire file.
# Submits stage 1 of the reduced reproduction, as approved by the supervisors on 2026-11-10:
#   - CIFAR-10/ResNet-56, seeds 0-5 (the "absolute minimum")
#   - CIFAR-100 and CINIC-10/ResNet-56, seed 0 each, to validate their runtimes before stage 2
#   - CIFAR-10/ResNet-56 seed 0 with the target trained at batch size 128 instead of 512
#     (controlled comparison; same data split, everything else unchanged)
# At most 3 GPU jobs run at once (supervisors' request; sciCORE is busy).
#
# Run from experiment/mia_comp/ on login12, after prepare_cinic10.py:
#   DRY_RUN=1 bash slurm/submit_stage1.sh    # show what would be done, submit nothing
#   bash slurm/submit_stage1.sh
# Afterwards: bash slurm/stage_report.sh
#
# The CIFAR-10 split and target model are copied from the pilot (pilot_l40s): same code and
# settings, so seed 0 here differs from the pilot's only in the LiRA loader workers, which
# is the consistency check for the loader change.
set -euo pipefail

export DATA_DIR="${DATA_DIR:-$HOME/mia_data_NOBACKUP}"
MAIN=miae_standard_exp
BS128=miae_bs128_exp
PILOT="$DATA_DIR/pilot_l40s/target"
CINIC_DIR="${CINIC_DIR:-$DATA_DIR/CINIC10}"
tasks=jobs/stage1.txt
jobs_file=slurm/stage1_jobs.txt
run() { if [ -n "${DRY_RUN:-}" ]; then echo "[dry run] $*" >&2; else "$@"; fi; }

[ -d "$CINIC_DIR/CINIC10_60ksubset" ] || { echo "CINIC-10 subset missing: run python experiment_scripts/prepare_cinic10.py" >&2; exit 1; }
[ -f "$PILOT/target_models/cifar10/resnet56/target_model_resnet56cifar10.pkl" ] || { echo "pilot target model missing in $PILOT" >&2; exit 1; }
mkdir -p logs jobs

# inputs: CIFAR-10 split (+ target model for the main run) from the pilot, CINIC-10 subset where
# get_cinic10() looks for it (--dataset_file_root = the experiment's target folder)
for exp in $MAIN $BS128; do
  run mkdir -p "$DATA_DIR/$exp/target"
  [ -e "$DATA_DIR/$exp/target/cifar10" ] || run cp -r "$PILOT/cifar10" "$DATA_DIR/$exp/target/"
done
if [ ! -e "$DATA_DIR/$MAIN/target/target_models/cifar10/resnet56" ]; then
  run mkdir -p "$DATA_DIR/$MAIN/target/target_models/cifar10"
  run cp -r "$PILOT/target_models/cifar10/resnet56" "$DATA_DIR/$MAIN/target/target_models/cifar10/"
fi
run ln -sfn "$CINIC_DIR/CINIC10_60ksubset" "$DATA_DIR/$MAIN/target/CINIC10_60ksubset"

# one line per seed set; the longest (CIFAR-100) first
cat > "$tasks" <<EOF
$MAIN cifar100 resnet56 0
$MAIN cinic10 resnet56 0
$MAIN cifar10 resnet56 0
$BS128 cifar10 resnet56 0
$MAIN cifar10 resnet56 1
$MAIN cifar10 resnet56 2
$MAIN cifar10 resnet56 3
$MAIN cifar10 resnet56 4
$MAIN cifar10 resnet56 5
EOF
n=$(wc -l < "$tasks")

submit() {  # submit <label> <sbatch args...>: prints the job ID and records it
  local label="$1"; shift
  if [ -n "${DRY_RUN:-}" ]; then echo "[dry run] sbatch $*" >&2; echo "DRYRUN_$label"; return; fi
  local id
  id=$(sbatch --parsable "$@") || { echo "sbatch failed for $label, stopping" >&2; exit 1; }
  echo "$label $id" >> "$jobs_file"
  echo "submitted $label: $id" >&2
  echo "$id"
}
[ -n "${DRY_RUN:-}" ] || : > "$jobs_file"

# target models: CIFAR-100 + CINIC-10 (splits + targets, batch 512), and CIFAR-10 at batch 128
prep_main=$(STEP=prepare EXP_NAME=$MAIN DATASETS="cifar100 cinic10" ARCHS=resnet56 \
  submit prepare_main -J mia_prep_main -p l40s --qos=l40s-6hours -t 02:00:00 --mem=16G --export=ALL slurm/pilot.sbatch)
prep_bs=$(STEP=prepare EXP_NAME=$BS128 DATASETS=cifar10 ARCHS=resnet56 TARGET_BATCH_SIZE=128 \
  submit prepare_bs128 -J mia_prep_bs128 -p l40s --qos=l40s-6hours -t 01:00:00 --mem=16G --export=ALL slurm/pilot.sbatch)
# %3 = at most 3 tasks at once; the two prepare jobs finish before any task starts
TASKS=$tasks submit seedsets --array="1-$n%3" --dependency="afterok:$prep_main:$prep_bs" --export=ALL slurm/seedset.sbatch >/dev/null

echo; echo "Stage 1: $n seed sets, max 3 GPUs at once. Watch: squeue --me    Report: bash slurm/stage_report.sh"

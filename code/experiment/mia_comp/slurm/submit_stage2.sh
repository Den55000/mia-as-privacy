#!/bin/bash
# [AI-generated: Claude Code, Claude Opus 5.5, 2026-11-11] Entire file.
# Submits stage 2 of the reduced reproduction (stage 1: slurm/submit_stage1.sh, REPRO_LOG.md):
#   1. image seed sets, ResNet-56: CIFAR-100 and CINIC-10 seeds 1-5, CIFAR-10 seeds 6-15
#      (+ optional extra seeds of the batch-size comparison, see BS128_SEEDS)
#   2. after those: Purchase100/Texas100 splits and MLP target models
#   3. after that: Purchase100/Texas100 seed sets, seeds 0-5 (6 attacks, no aug)
# The three parts run one after another, each with at most 3 GPU jobs at once, so the
# whole stage never uses more than 3 GPUs (supervisors' request).
#
# Run from experiment/mia_comp/ on login12. The tabular data must be in data/purchase100-texas100
# (git clone https://github.com/xehartnort/Purchase100-Texas100-datasets.git data/purchase100-texas100;
# compute nodes have no internet, so the code's own download would fail there).
#   DRY_RUN=1 bash slurm/submit_stage2.sh            # show what would be done, submit nothing
#   bash slurm/submit_stage2.sh
#   BS128_SEEDS="1 2" bash slurm/submit_stage2.sh    # also extra batch-size-comparison seeds
# Afterwards: bash slurm/stage_report.sh slurm/stage2_jobs.txt
set -euo pipefail

export DATA_DIR="${DATA_DIR:-$HOME/mia_data_NOBACKUP}"
MAIN=miae_standard_exp
BS128=miae_bs128_exp
mlp=mlp_for_texas_purchase
img_tasks=jobs/stage2_images.txt
tab_tasks=jobs/stage2_tabular.txt
jobs_file=slurm/stage2_jobs.txt

for f in purchase100 texas100; do
  [ -f "data/purchase100-texas100/$f.npz" ] || { echo "data/purchase100-texas100/$f.npz missing, see the header" >&2; exit 1; }
done
for d in cifar10 cifar100 cinic10; do
  [ -f "$DATA_DIR/$MAIN/target/target_models/$d/resnet56/target_model_resnet56$d.pkl" ] || { echo "$d target model missing: run stage 1 first" >&2; exit 1; }
done
if [ -n "${BS128_SEEDS:-}" ]; then
  [ -f "$DATA_DIR/$BS128/target/target_models/cifar10/resnet56/target_model_resnet56cifar10.pkl" ] || { echo "batch-128 target missing" >&2; exit 1; }
fi
mkdir -p logs jobs

# image seed sets, the longest (CIFAR-100) first
{
  for sd in 1 2 3 4 5; do echo "$MAIN cifar100 resnet56 $sd"; done
  for sd in 1 2 3 4 5; do echo "$MAIN cinic10 resnet56 $sd"; done
  for sd in 6 7 8 9 10 11 12 13 14 15; do echo "$MAIN cifar10 resnet56 $sd"; done
  for sd in ${BS128_SEEDS:-}; do echo "$BS128 cifar10 resnet56 $sd"; done
} > "$img_tasks"
{
  for d in purchase100 texas100; do for sd in 0 1 2 3 4 5; do echo "$MAIN $d $mlp $sd"; done; done
} > "$tab_tasks"
n_img=$(wc -l < "$img_tasks"); n_tab=$(wc -l < "$tab_tasks")

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

img=$(TASKS=$img_tasks submit seedsets_images --array="1-$n_img%3" --export=ALL slurm/seedset.sbatch)
# afterany: the tabular part also runs if an image task failed (that one can be resubmitted alone)
prep=$(STEP=prepare EXP_NAME=$MAIN DATASETS="purchase100 texas100" ARCHS=$mlp \
  submit prepare_tabular -J mia_prep_tab -p l40s --qos=l40s-6hours -t 02:00:00 --mem=32G \
  --dependency="afterany:$img" --export=ALL slurm/pilot.sbatch)
TASKS=$tab_tasks submit seedsets_tabular --array="1-$n_tab%3" --mem=32G \
  --dependency="afterok:$prep" --export=ALL slurm/seedset.sbatch >/dev/null

echo; echo "Stage 2: $n_img image + $n_tab tabular seed sets, max 3 GPUs at once."
echo "Watch: squeue --me    Report: bash slurm/stage_report.sh $jobs_file"

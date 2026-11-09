#!/bin/bash
# [AI-generated: Claude Code, Claude Opus 5.5, 2026-10-23] Entire file.
# Smoke test for the pipeline (plan step B4). Run from experiment/mia_comp/ with the venv active.
#
#   bash experiment_scripts/smoke_test.sh quick   # ~75 min: all 7 attacks, 2 epochs each.
#                                                 # Checks the code runs; the AUCs are meaningless.
#   bash experiment_scripts/smoke_test.sh full    # ~1-2 h: paper settings, LOSS + calibration.
#                                                 # Gives real timings and AUCs to compare with Table 1.
#   bash experiment_scripts/smoke_test.sh lira    # after "full": adds LiRA for seed 0 (many hours on a laptop).
#
# Results go to $DATA_DIR/smoke_quick or $DATA_DIR/smoke_full, never to the main
# experiment folder. A timing log is written to <that folder>/timings.log.
set -euo pipefail

mode="${1:-quick}"
export DATA_DIR="${DATA_DIR:-$(realpath ../../../data)}"
export DATASETS="cifar10" ARCHS="resnet56"

case "$mode" in
  quick)
    export EXP_NAME="smoke_quick" EPOCHS=2
    export MIAS="lira reference shokri losstraj calibration yeom aug"
    ;;
  full)
    export EXP_NAME="smoke_full"
    unset EPOCHS
    export MIAS="yeom calibration"
    ;;
  lira)
    export EXP_NAME="smoke_full"
    unset EPOCHS
    export MIAS="lira"
    ;;
  *)
    echo "usage: $0 quick|full|lira"; exit 1 ;;
esac

out="$DATA_DIR/$EXP_NAME"
mkdir -p "$out"
timings="$out/timings.log"

python -c "import torch; assert torch.cuda.is_available(), 'no GPU visible'; print('GPU:', torch.cuda.get_device_name(0))"

timed() {  # timed <label> <command...>: run a step and append its wall-clock time to the log
  local label="$1"; shift
  local start=$SECONDS
  "$@"
  echo "$(date '+%F %T') $mode $label $(( (SECONDS - start) / 60 )) min $(( (SECONDS - start) % 60 )) s" | tee -a "$timings"
}

if [ -f "$out/target/target_models/cifar10/resnet56/target_model_resnet56cifar10.pkl" ]; then
  echo "Target model exists, skipping prepare_target"
else
  timed "prepare_target" bash experiment_scripts/prepare_target.sh
fi
for mia in $MIAS; do
  if [ -f "$out/preds_sd0/cifar10/resnet56/$mia/pred_$mia.npy" ]; then
    echo "pred_$mia.npy exists, skipping"; continue
  fi
  MIAS="$mia" timed "attack_$mia" bash experiment_scripts/obtain_pred.sh 0
done

python experiment_scripts/check_preds.py "$out" 0
echo "Timings: $timings"

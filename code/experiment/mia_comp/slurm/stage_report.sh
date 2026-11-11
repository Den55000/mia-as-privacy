#!/bin/bash
# [AI-generated: Claude Code, Claude Opus 5.5, 2026-11-10] Entire file.
# Summarises stage 1 (slurm/submit_stage1.sh): job states and resources, runtimes per dataset,
# GPU use, the paper-style checks, the loader-change consistency check and the 512 vs 128
# batch-size comparison. Run from experiment/mia_comp/: bash slurm/stage_report.sh
DATA_DIR="${DATA_DIR:-$HOME/mia_data_NOBACKUP}"
source ../../.venv/bin/activate
MAIN="$DATA_DIR/miae_standard_exp"
BS128="$DATA_DIR/miae_bs128_exp"
ids=$(awk '{print $2}' slurm/stage1_jobs.txt | paste -sd,)

echo "== SLURM accounting"
sacct -j "$ids" --format=JobID%16,JobName%16,State,Elapsed,MaxRSS | grep -vE '\.(extern|0)\b'

echo; echo "== Runtimes (timings.log; seconds)"
for exp in "$MAIN" "$BS128"; do
  [ -f "$exp/timings.log" ] || continue
  echo "-- $(basename "$exp")"
  # per seed set: total over all attacks
  awk '/_attack_/ {split($5, a, "_attack_"); t[a[1]] += $6} END {for (k in t) printf "  %-26s %6.2f h\n", k, t[k]/3600}' "$exp/timings.log" | sort
  for f in "$exp"/gpu_*.csv; do
    [ -f "$f" ] && awk -F', ' 'NR>1 {u+=$2; if ($3+0>m) m=$3+0; n++} END {if (n) printf "  %s: mean GPU util %.0f%%, peak mem %d MiB\n", FILENAME, u/n, m}' "$f"
  done
done

echo; echo "== Checks against the paper (check_preds.py)"
for sd in 0 1 2 3 4 5; do
  [ -d "$MAIN/preds_sd$sd" ] && { echo "-- miae_standard_exp seed $sd"; python experiment_scripts/check_preds.py "$MAIN" "$sd" | tail -n +2; }
done
[ -d "$BS128/preds_sd0" ] && { echo "-- miae_bs128_exp seed 0"; python experiment_scripts/check_preds.py "$BS128" 0; }

c10=cifar10/resnet56
mem="$MAIN/target/cifar10/attack_set_membership.npy"
echo; echo "== Loader change: pilot seed 0 (0 workers) against stage-1 seeds 0-5 (8 workers)"
echo "   Consistent if the pilot's AUC/TPR lie within the seeds' spread. Seed 0 shares the pilot's"
echo "   target model and seed, so its non-LiRA attacks (unaffected by the change) show pure GPU nondeterminism."
args=("pilot_sd0_w0=$DATA_DIR/pilot_l40s/preds_sd0/$c10")
for sd in 0 1 2 3 4 5; do [ -d "$MAIN/preds_sd$sd/$c10" ] && args+=("main_sd${sd}_w8=$MAIN/preds_sd$sd/$c10"); done
python experiment_scripts/compare_preds.py "$mem" "${args[@]}"

echo "== Batch size: target at 512 (main) against 128, seed 0, same split"
python experiment_scripts/compare_preds.py "$mem" "bs512_sd0=$MAIN/preds_sd0/$c10" "bs128_sd0=$BS128/preds_sd0/$c10"

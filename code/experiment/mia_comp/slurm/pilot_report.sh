#!/bin/bash
# [AI-generated: Claude Code, Claude Opus 5.5, 2026-11-09] Entire file.
# Summarises the pilot (PLAN.md, D5 steps 3-4). Run from experiment/mia_comp/: bash slurm/pilot_report.sh
DATA_DIR="${DATA_DIR:-$HOME/mia_data_NOBACKUP}"
source ../../.venv/bin/activate   # for check_preds.py below (the login shell has no venv)
ids=$(awk '{print $2}' slurm/pilot_jobs.txt | paste -sd,)

echo "== SLURM accounting (Elapsed = wall time; MaxRSS -> --mem; TotalCPU/Elapsed -> cores used)"
sacct -j "$ids" -X --format=JobName%16,Partition,State,Elapsed,AllocCPUS,TotalCPU,ReqMem
sacct -j "$ids" --format=JobID%14,MaxRSS | grep -E 'batch' || true

echo; echo "== seff"
for id in ${ids//,/ }; do seff "$id" | grep -E 'Job ID|State|CPU Efficiency|Memory Utilized|Job Wall'; echo; done

for exp in "$DATA_DIR"/pilot_*; do
  echo "== $(basename "$exp")"
  cat "$exp/timings.log" 2>/dev/null
  for f in "$exp"/gpu_*.csv; do
    [ -f "$f" ] && awk -F', ' 'NR>1 {u+=$2; if ($3+0>m) m=$3+0; n++} END {if (n) printf "%s: mean GPU util %.0f%%, peak mem %d MiB\n", FILENAME, u/n, m}' "$f"
  done
  # per-shadow-model time from the checkpoint timestamps (useful for the killed WRN probe)
  find "$exp" -name shadow.pth -printf '%T@\n' 2>/dev/null | sort -n |
    awk 'NR>1 {d=$1-p; s+=d; n++} {p=$1} END {if (n) printf "LiRA: %d shadows done, %.1f min per shadow -> 20 shadows = %.1f h\n", n+1, s/n/60, 20*s/n/3600}'
  python experiment_scripts/check_preds.py "$exp" 0 2>/dev/null | tail -12
done

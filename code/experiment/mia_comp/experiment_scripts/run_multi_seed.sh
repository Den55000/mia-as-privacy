#!/bin/bash
# [AI-generated: Claude Code, Claude Opus 5.5, 2026-10-23] Changes to the authors' script: settings via environment variables, stop at first error, per-run scratch folders, sequential seeds. See docs/REPRO_LOG.md.
# Run obtain_pred.sh for several seeds, one after another.
# Run from experiment/mia_comp/:  bash experiment_scripts/run_multi_seed.sh {0..5}
# DATASETS / ARCHS / MIAS / DATA_DIR are passed through to obtain_pred.sh.
#
# Changed from the original, which ran all seeds in parallel on CUDA device 1 (fixes #2, #3).
# Set PARALLEL=1 to restore the parallel behaviour on a machine with enough GPU memory.

seeds=("$@")

export DATA_DIR="${DATA_DIR:-$(realpath ../../../data)}"
script_out_dir="$DATA_DIR"
mkdir -p "$script_out_dir"

for sd in "${seeds[@]}"; do
    log_file="${script_out_dir}/output_${sd}.log"
    echo "Seed $sd -> $log_file"
    if [ "${PARALLEL:-0}" = "1" ]; then
        bash experiment_scripts/obtain_pred.sh "$sd" > "$log_file" 2>&1 &
    else
        bash experiment_scripts/obtain_pred.sh "$sd" > "$log_file" 2>&1 || echo "Seed $sd FAILED, see $log_file"
    fi
done

wait
echo "All tasks completed. Check output files in $script_out_dir"

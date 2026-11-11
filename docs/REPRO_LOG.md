# Reproduction log

Records every change to the authors' code, the commands run, wall-clock times and errors (see PLAN.md, B3).
Code: https://github.com/RPI-DSPlab/mia-disparity at commit `71a2965`, vendored in `code/`.

## Environment (laptop)

- GPU: NVIDIA GeForce RTX 5070 Laptop GPU (8 GB, sm_120), 30 GB RAM, Arch Linux.
- The repo pins PyTorch 2.0.1 + CUDA 11.8, which does not support this GPU. Using PyTorch with CUDA 12.8 in `code/.venv` (Python 3.11), see PLAN.md B2.
- The default `uv` HTTP timeout fails on the ~9 GB PyTorch download; `UV_HTTP_TIMEOUT=600` works.

## Code changes

See PLAN.md B3, "Prepared changes", for the full list. Summary:

| File | Change | Affects results? |
|---|---|---|
| `experiment_scripts/prepare_target.sh`, `obtain_pred.sh`, `run_multi_seed.sh` | Settings via environment variables (paper's values as defaults), stop at first error, one scratch folder per seed/dataset/arch/attack, seeds run one after another | No |
| `obtain_pred.py` | Skip the model-compatibility check for `--save_dataset` (crashed; the original script hid this by continuing after errors) | No |
| `miae/utils/dataset_utils.py` | `torch._utils._accumulate` → `itertools.accumulate` (removed in newer PyTorch; same split) | No |
| `miae/attacks/aug_mia.py` | `scipy.ndimage.shift` in chunks of 1,000 images instead of all at once (~34 GB for 30k images). Output identical: max abs diff 0.0 for all 9 shifts | No |
| new: `experiment_scripts/smoke_test.sh`, `check_preds.py` | Smoke test driver and prediction checks | n/a |

Bugs found but **not fixed** (they change results; reported in the review):
- `aug_mia.py`, `prepare()`: member labels come from a `DataLoader` with `shuffle=True`, features in dataset order, so labels and features are mismatched for members.
- `aug_mia.py`, `create_translates()`: shifts `(0, dy, dx, 0)` assume (N, H, W, C), but images are (N, C, H, W). Verified: `(0, ±2, 0, 0)` replaces all three channels with the red/blue channel; width is never shifted.

## 2026-10-23: quick smoke test

Command (from `code/experiment/mia_comp/`): `bash experiment_scripts/smoke_test.sh quick`
CIFAR-10, ResNet-56, seed 0, all 7 attacks, `EPOCHS=2`. Output in `data/smoke_quick/`.

Timeline (from `timings.log`, file timestamps and `journalctl`):

| Time | Event |
|---|---|
| 16:52:56 | Data splits and target model done (32 s) |
| 16:52:56–17:03:26 | LiRA: 20 shadow models trained (~31 s each) |
| **17:06:11–19:01:15** | **Laptop suspended (115 min), in the middle of LiRA's queries** |
| 19:19:01 | LiRA done. `timings.log` says 146 min 5 s; **without the pause: 31.0 min** |
| 19:21:39 | Reference done (2 min 38 s) |
| 19:22:15 | Class-NN done (36 s) |
| 19:23:33 | Loss trajectory done (1 min 18 s) |
| 19:28:02 | Calibration done (4 min 29 s) |
| 19:28:12 | LOSS done (10 s) |
| 19:28–20:09 | Augmentation, first attempt (before the chunked-shift fix): **killed by the kernel OOM killer** at 20:09:29 (python3, ~27.8 GB RSS) |
| 20:23:16 | Restarted with the fix; earlier attacks skipped (predictions exist). `run.log` only holds this second run |
| 20:56:11 | Augmentation done (32 min 55 s). All checks passed |

Total compute time without the pause and the failed attempt: **~74 min**.

Results (`check_preds.py`): all attacks give shape (30000,), AUC 0.498–0.503, TPR@1% FPR 0.007–0.011, TPR@0.1% FPR 0.001, i.e. random guessing, as expected for a target model at 23% train / 24% test accuracy after 2 epochs.

Notes:
- `timed` in `smoke_test.sh` measures wall-clock time, so suspends are counted. Disable suspend (or use `systemd-inhibit`) for long runs.
- `run.log` was overwritten by the restart, so the console output of the first attempt is lost. Append (`>>`) or use one log per attempt next time.

## 2026-11-09/10: sciCORE pilot and follow-up checks

Pilot (PLAN.md D5): CIFAR-10/ResNet-56 seed 0, all 7 attacks, on L40S and A100, plus a 6 h CIFAR-100/WRN-32-4 LiRA probe; ~22.5 GPU-hours including four follow-up jobs. AUCs within 0.02 of Table 1. One seed set took 5.5 h (L40S) / 9.3 h (A100), against 11.7 h estimated. Full write-up: the "sciCORE Pilot Report" doc.

Findings:
- **Data loading is the bottleneck.** GPU use 11–30%, about one CPU core busy. LiRA's shadow-model `DataLoader`s (`miae/attacks/lira_mia.py`) have no `num_workers`, so loading and augmentation run in the main process. Benchmark (`slurm/bench_shadow.sbatch`, 2 shadow models per setting, same node): 11.2 → 4.9 min per shadow model on L40S, 20.7 → 9.4 min on A100 with 8 workers. Shadow-model accuracy unchanged (99.5–99.6% train, 88.7–88.8% test).
- **Batch-size mismatch between target and shadow models.** `obtain_pred.py` trains the target at `--batch_size` 512; LiRA reads `shadow_batchsize` (default 128) and ignores the `batch_size` it is passed. The paper (App. A.1) says target and shadow models are trained the same way and gives no batch size. Effect: CIFAR-100/WRN-32-4 target at 512 reached 67% train / 40% test accuracy (clean images), at 128 it reached 98% / 54% (`slurm/eval_target.sbatch`, `slurm/retrain_target.sbatch`). CIFAR-10/ResNet-56 at 512 is fine (96% / 80%) and matches the paper's AUCs.

## Code changes for the main run (approved by the supervisors 2026-11-10)

| File | Change | Affects results? |
|---|---|---|
| `miae/attacks/lira_mia.py`, `obtain_pred.py`, `experiment_scripts/obtain_pred.sh` | Optional data-loader workers for LiRA shadow-model training (`SHADOW_WORKERS` → `--shadow_num_workers` → `num_workers`, default 0 = authors' behaviour). Main run uses 8 | Not the method; the augmentation random streams differ, so results are not bit-identical. Consistency checked in stage 1 (`compare_preds.py`: pilot seed 0 with 0 workers against seeds 0–5 with 8) |
| `experiment_scripts/prepare_target.sh` | Optional `TARGET_BATCH_SIZE` (default 512 = authors' setting). Main run keeps 512; a controlled CIFAR-10 seed-0 run at 128 quantifies the effect | Only when set |
| new: `experiment_scripts/prepare_cinic10.py` | Script version of `process_CINIC10.ipynb` (download + 3,000 images per class). Listings sorted before sampling, because `os.listdir` order is file-system dependent, so the authors' exact subset cannot be reproduced anyway | Which CINIC-10 images are used (unavoidable) |
| new: `slurm/seedset.sbatch`, `slurm/submit_stage1.sh`, `slurm/stage_report.sh`, `experiment_scripts/compare_preds.py` | Stage-1 run: one array task per seed set, at most 3 GPU jobs at once (supervisors' request); removes incomplete LiRA shadow-model folders before a resubmitted task (fix #12) | No |
| `slurm/pilot_report.sh` | Activates the venv, so `check_preds.py` prints | No |

The main run (`miae_standard_exp`) reuses the pilot's CIFAR-10 data split and target model (`pilot_l40s`, same code and settings), so its seed 0 differs from the pilot's only in the loader workers.

## 2026-11-10/11: stage 1 of the reduced reproduction (sciCORE, L40S)

Command (from `code/experiment/mia_comp/` on login12, code at commit `a71f521`): `bash slurm/submit_stage1.sh`, report: `bash slurm/stage_report.sh`.
Jobs: prepare 30997054 (CIFAR-100 + CINIC-10 splits and targets), 30997055 (CIFAR-10 target at batch size 128), array 30997056_[1-9%3] (one seed set per task, at most 3 GPUs at once). All COMPLETED, 2026-11-10 18:03 to 2026-11-11 05:34. Results in `~/mia_data_NOBACKUP/miae_standard_exp` and `miae_bs128_exp`. LiRA shadow models trained with 8 loader workers.

Cost: ~31.4 GPU-hours (estimate 33). With the pilot (~22.5): ~54 GPU-hours. Peak 4.4 GB RAM, 5.4 GB GPU memory, mean GPU use 34–45%.

Runtime per seed set (all 7 attacks, ResNet-56, L40S):

| Dataset | Seeds | Hours per seed set | vs. CIFAR-10 |
|---|---|---|---|
| CIFAR-10 | 0–5 | 3.17–3.34 | 1.0× |
| CIFAR-100 | 0 | 4.58 (LiRA 166 min, aug 70, losstraj 20) | 1.4× (1.7× assumed) |
| CINIC-10 | 0 | 3.65 (LiRA 105 min, aug 71, losstraj 22) | 1.1× |

Target models (train / test accuracy, batch size 512): CIFAR-10 96% / 80% (the pilot's model, reused), CIFAR-100 96% / 47%, CINIC-10 81% / 61%. The paper reports none of these.

Single-instance AUC, seed 0, against Table 1 (paper / ours):

| Attack | CIFAR-10 | CIFAR-100 | CINIC-10 |
|---|---|---|---|
| losstraj | 0.635 / 0.648 | 0.852 / 0.856 | 0.673 / 0.645 |
| reference | 0.608 / 0.606 | 0.813 / 0.821 | 0.615 / 0.612 |
| lira | 0.585 / 0.583 | 0.802 / 0.811 | 0.601 / 0.598 |
| calibration | 0.603 / 0.613 | 0.721 / 0.725 | 0.623 / 0.635 |

CIFAR-10 seeds 0–5 (AUC range): losstraj 0.584–0.653, reference 0.606–0.609, lira 0.577–0.587, calibration 0.613–0.622, aug 0.570–0.587, shokri 0.568–0.575, LOSS 0.553 (deterministic). CINIC-10 losstraj is 0.028 below the paper; our CINIC-10 image subset necessarily differs from the authors' (see `prepare_cinic10.py`).

**Loader-change consistency check** (`compare_preds.py`, pilot seed 0 with 0 workers against stage-1 seeds 0–5 with 8): pilot LiRA AUC 0.586 / TPR@1%FPR 0.017 lies within the seeds' 0.577–0.587 / 0.016–0.021; Reference likewise (0.607 / 0.041 against 0.606–0.609 / 0.039–0.045). The pipeline is deterministic: on seed 0, all attacks that do not use LiRA's shadow models reproduced the pilot's predictions exactly (Jaccard 1.00 of detected members, identical AUC). Overlap of LiRA's detected members between pilot and seed 0 (0.17) is the same as between different seeds (0.13–0.16). Conclusion: consistent.

**Batch-size comparison** (CIFAR-10 seed 0, same split; target at 512 as in the authors' code vs. 128 as their shadow models): target 96% / 80% → 99% / 84%.

| Attack | AUC 512 → 128 | TPR@1%FPR 512 → 128 |
|---|---|---|
| lira | 0.583 → 0.645 | 0.018 → 0.060 |
| reference | 0.606 → 0.669 | 0.040 → 0.075 |
| aug | 0.581 → 0.600 | 0.007 → 0.008 |
| LOSS | 0.553 → 0.581 | 0.000 → 0.000 |
| losstraj | 0.648 → 0.647 | 0.044 → 0.044 |
| calibration | 0.613 → 0.615 | 0.026 → 0.027 |
| shokri | 0.571 → 0.578 | 0.011 → 0.009 |

The two shadow-model attacks that assume shadows are trained like the target (LiRA, Reference) gain ~0.06 AUC, about 6× the seed-to-seed spread; the others barely change. The paper's Table 1 values for LiRA and Reference match the 512 setting, so its attack ranking likely reflects the mismatch. One seed only.

Open: losstraj on CIFAR-10 seed 3 has AUC 0.584 (other seeds 0.641–0.653), while its TPR at 1% / 0.1% FPR is normal (0.040 / 0.011). Its shadow and distilled models trained normally (same accuracies as seed 1), so the difference comes from the final attack classifier, which is not logged. Treated as instance variation; check whether it recurs in stage 2.

Fixed afterwards: `slurm/stage_report.sh` summed the wrong `timings.log` columns (showed 0.00 h).

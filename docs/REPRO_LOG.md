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

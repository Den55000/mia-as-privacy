# Plan: Understand and reproduce "Membership Inference Attacks as Privacy Tools: Reliability, Disparity and Ensemble"

Paper: Wang, Zhang, Chen, Baracaldo, Kadhe, Yu. ACM CCS 2025, https://doi.org/10.1145/3719027.3744818 (a local copy of the PDF is kept in `docs/` but not published).
Extended version with the full appendix: arXiv 2506.13972.
Code: https://github.com/RPI-DSPlab/mia-disparity (checked at commit `71a2965`, Jan 2026).

The plan has three parts:

- **Part A**: learn the topic, so the experiments make sense.
- **Part B**: set up the code and reproduce the main results, starting small.
- **Part C**: questions to probe for the review.

---

## 0. The paper in five sentences

1. Membership inference attacks (MIAs) guess whether a sample was in a model's training set. They are widely used to measure privacy (auditing, unlearning checks, defense benchmarks).
2. The paper shows that **two runs of the same attack with different random seeds** detect very different sets of members (*instance-level disparity*, measured by the **consistency** score, a mean pairwise Jaccard index).
3. **Different attacks** at the same FPR also detect largely different members (*method-level disparity*). This is measured by the Jaccard similarity of their **coverage** (the union of true positives over seeds) and **stability** (the intersection of true positives over seeds).
4. So relying on one attack, or one run of one attack, underestimates privacy risk.
5. They propose **ensembles**. First combine the instances of each attack (AND = stability, OR = coverage, or majority vote), then OR across attacks. On CIFAR-10 this gives about 5× higher TPR at 0.1% FPR than the best single attack.

---

## Part A: Understand the topic

### A1. Background, in this order (about 1–2 days)

| # | Read | Why |
|---|------|-----|
| 1 | Shokri et al. 2017, *Membership Inference Attacks against ML Models* (ref [44]) | The original shadow-model idea. Its attack is "Class-NN" (`shokri` in the code). |
| 2 | Yeom et al. 2018 (ref [53]) | The simplest attack, a loss threshold (`yeom` / "LOSS"). Links MIAs to overfitting. |
| 3 | Carlini et al. 2022, *MIAs From First Principles* (ref [3]), sections 1–4 | LiRA, and **why TPR at low FPR matters more than AUC**. This is the metric the paper builds on. |
| 4 | Skim: Watson et al. 2022 (difficulty calibration, [51]), Ye et al. 2022 (Reference attack, [52]), Liu et al. 2022 (Loss Trajectory, [30]), Choquette-Choo et al. 2021 (label-only augmentation, [8]) | The other four attacks. You only need to know *what signal each one uses*. |
| 5 | Optional: Aerni et al. 2024, *Evaluations of ML Privacy Defenses are Misleading* ([1]) | Related critique of how MIAs are used for evaluation. Useful for positioning the review. |

### A2. Cheat sheet: the 7 attacks

| Paper name | Code name | Signal | Shadow models | Randomness from | Relative cost |
|---|---|---|---|---|---|
| LOSS | `yeom` | target loss vs. global threshold | 0 | none (consistency = 1) | trivial |
| Class-NN | `shokri` | full logit vector → per-class MLP | 10, on **disjoint** aux subsets | shadows + attack MLPs | low (small shadows) |
| Augmentation | `aug` | predictions on 5 rotated versions → MLP | 1 | shadow + MLP | low |
| Difficulty Calibration | `calibration` | target loss − shadow loss | 1 | shadow, aux split | low |
| LiRA (online) | `lira` | per-sample Gaussian fits of IN/OUT losses | 20, each on half of aux | aux split, training | **high** |
| Reference (Attack R) | `reference` | likelihood ratio vs. reference models | **reuses LiRA's 20** | same as LiRA | free once LiRA has run |
| Loss Trajectory | `losstraj` | loss across 100 distillation epochs → MLP | 1 shadow + 2 distillations | shadow, distillation, MLP | medium-high |

The ensemble (Section 5) uses only **lira, reference, losstraj, calibration**. These are the four whose precision improves under multi-instance stability (Fig. 9b).

### A3. The paper's machinery

Work through each definition by hand on a toy example before running anything.

- **Setup (Alg. 2):** dataset D is split 50/50 into target D_T and auxiliary D_A. D_T is split 50/50 into members (the target model's training set) and non-members. CIFAR-10: 30k aux, 15k members, 15k non-members.
- **AdjustFPR (Alg. 1):** each instance's scores are thresholded so that its FPR equals β (mainly β = 0.1). The threshold uses the ground-truth labels. This is an evaluation device, not something a real attacker can do. Keep this in mind (see Part C).
- **Consistency (Eq. 6):** the mean Jaccard index over all pairs of instances' predicted-member sets.
- **Coverage (Eq. 7) / Stability (Eq. 8):** the union / intersection of true positives across the |S| seeds. The code calls them *union* / *intersection*, and calls instances *seeds*.
- **Method-level disparity:** the Jaccard index between two attacks' coverage sets (or stability sets) (Fig. 10).
- **𝒜-unique samples (Eq. 9):** members in 𝒜's stability set that are in no other attack's stability set. PCA is run on the target model's logits for these (Fig. 12).
- **Top-x Class-NN (Sec. 4.5.2):** Class-NN restricted to the top-x logits, to isolate the effect of the signal (Fig. 13).
- **Ensembles (Eqs. 11–16):** step 1 combines instances within an attack (AND / OR / majority), step 2 ORs across attacks. To get a ROC curve, sweep the base FPR β over 100 log-spaced values from 1e-6 to 1.

### A4. Hands-on warm-up (half a day, no GPU)

Write a small numpy notebook (e.g. `notebooks/00_toy_metrics.ipynb`) that:

1. Simulates 3 "attacks" × 6 "seeds" of scores for 1000 members and 1000 non-members (e.g. member score = signal + noise, where the noise is redrawn per seed).
2. Implements `adjust_fpr(gt, scores, beta)` exactly as in Alg. 1 (`sklearn.metrics.roc_curve`, then `argmin(|FPR-β|)`).
3. Computes consistency, coverage, stability and the pairwise Jaccard index.
4. Implements the stability, coverage and majority ensembles and plots their ROC curves.

Payoff: you will know what every number in the paper means. You will also have an independent implementation to check against the authors' `miae/eval_methods/prediction.py`.

**Null baseline for the review:** compute the Jaccard index of two *independent* random detectors with the same TPR p. It is about p² / (2p − p²); for p = 0.2 that is about 0.11. Compare this with the paper's consistency values (< 0.4) and the stability Jaccard values (< 0.1). It shows how much of the "disparity" is beyond chance.

### A5. Self-check questions before Part B

- Why does coverage TPR go up while its FPR also goes up (Fig. 7)? Why does stability push FPR toward 0 (Fig. 8)?
- Why is the LOSS attack's consistency exactly 1?
- Why does Class-NN have particularly low consistency? (Hint: disjoint shadow sets.)
- Why does the ensemble need a sweep over base FPRs to draw a ROC curve, instead of a score threshold?
- What does a stability Jaccard of 0.00 between LiRA and Class-NN (Fig. 10b) actually mean?

---

## Part B: Reproduce the experiments

### B0. Compute check (do this first)

Your machine has an **RTX 5070 Laptop GPU (8 GB, Blackwell / sm_120)**, 24 CPU threads, 30 GB RAM, and about 570 GB free disk.

- The full paper grid is 5 datasets × 4 architectures × 7 attacks × 6–16 seeds. That is **not feasible** on one laptop GPU. The authors report about 3500 GPU-minutes just for 6 instances of the 4 ensemble attacks on CIFAR-10/ResNet-56 (Fig. 17).
- **Realistic target:** CIFAR-10 + ResNet-56 (the paper's main setting), all 7 attacks, 6 seeds. Expect a rough order of **tens of GPU-hours**, mostly LiRA's 20 shadow models per seed. Measure the real number in the smoke test (B4) before committing.
- Optional extras, if time allows: 16 seeds (Figs 7–9 use 16), and CIFAR-100 (most of the paper's contrast between datasets).

#### B0b. Running on university GPUs instead

> For the full-paper reproduction on sciCORE, see **Part D**. It has the exact job list, partitions and walltimes. The overview below is kept for reference.

Cost reference: the paper's appendix C.1 timings on an **Nvidia L40S**, with ResNet-56 on CIFAR-10. One seed (instance) costs:

- LiRA: 580 min, of which 540 min is training its 20 shadow models.
- Reference: reuses LiRA's shadow models, so it costs little extra.
- Loss Trajectory: 17 min.
- Calibration: 5 min.

Adding Class-NN, Aug, LOSS and the target model, **one seed of all 7 attacks is about 11 GPU-hours**. CIFAR-100 trains for 100 epochs instead of 60, so it is about 18 GPU-hours per seed.

| Tier | What | Paper items | GPU-hours (L40S-class) |
|---|---|---|---|
| 1 | CIFAR-10, ResNet-56, 7 attacks, 6 seeds | Figs 1, 2, 4a (ResNet bars), 10, 11a/b, 12, 15, 16, 17; Table 1 CIFAR-10 column | **≈ 70** |
| 2 | Tier 1, plus 10 more CIFAR-10 seeds (16 total), plus CIFAR-100 and CINIC-10 with 6 seeds each | Adds Figs 7–9, the full Table 1, Figs 4b/4c (ResNet), 11c/d | **≈ 350** (≈ 530 with 16 seeds on CIFAR-100 too) |
| 3 | Everything: 4 architectures × 3 image datasets, 4 repeated data partitions (Fig. 10 averages 4), Figs 5/6 (up to 100 shadow models), Fig. 13, Purchase/Texas | Full paper and appendix | **≈ 2,500–3,000** (Fig. 6 alone is about 800) |

Each seed is an independent job, so N GPUs cut wall-clock time by about N. One seed still takes about 11 h, because LiRA trains its shadow models one after another.

**Per job:** 1 GPU with **≥ 16 GB** memory (ResNet-56 at batch 512 needs only a few GB; WRN-32-4 and VGG-16 need more). Also about **10 CPU cores** and **32 GB RAM**. Measured in the smoke test: with the chunked shift fix (B3), the Augmentation attack stays at a few GB. Without it, it needs ~34 GB, so use `--mem=48G` if you run the unmodified code. The authors don't report their CPU or RAM; these figures are inferred from the code, which uses at most 1 main process + 8 data-loader workers, and holds the CIFAR data in memory as roughly 1–2 GB of tensors. The 27 min per shadow model on an L40S suggests the jobs are limited by data loading (`num_workers=2` in the code), so extra CPU cores plus a higher `num_workers` may speed things up a lot.

**Storage (disk):** about **20 GB** for Tier 1 and about **100 GB** for the full paper. Both figures include the Python environment (~5–10 GB) and margin. These are estimates from what the code saves:

- Dataset split pickles: ~1 GB per dataset.
- LiRA's 20 shadow checkpoints: kept, ~70 MB per seed for ResNet-56 and ~1.2 GB for VGG-16.
- Loss Trajectory's 200 per-epoch distillation checkpoints: temporary and deleted after each attack, ~0.7 GB for ResNet-56 and ~12 GB for VGG-16. Every job running at the same time needs this space.
- Predictions: tiny.

Put `DATA_DIR` on the cluster's scratch filesystem, and copy `preds_sd*/` and the figures to backed-up storage, because scratch is usually purged.

**Cluster notes:**
- Download CIFAR on the login node first, because compute nodes often have no internet access.
- Cluster GPUs (V100/A100/L40S/H100) work with the repo's **original** `miae_env.yml` (PyTorch 2.0.1 + CUDA 11.8, driver ≥ 520). That is more faithful than the laptop environment in B2 and makes fix #4 in B3 unnecessary.
- Turn `obtain_pred.sh <seed>` into a SLURM array job with one seed per task.

### B1. Get the code

```bash
cd ~/Projects/mia_as_privacy
git clone https://github.com/RPI-DSPlab/mia-disparity.git code
cd code && git checkout 71a2965      # pin the version this plan was written against
```

Read `README.md` and `experiment/mia_comp/README.md`. Then skim `experiment/mia_comp/obtain_pred.py` (the main runner) and `miae/attacks/*.py`.

### B2. Environment

The repo's `miae_env.yml` pins **PyTorch 2.0.1 + CUDA 11.8**. That build **does not support Blackwell GPUs**: you will get "no kernel image is available" errors. Use a fresh environment with a recent PyTorch built for CUDA 12.8 instead:

```bash
cd ~/Projects/mia_as_privacy/code
uv venv --python 3.11 .venv          # or: conda create -n miae python=3.11
source .venv/bin/activate
UV_HTTP_TIMEOUT=600 uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128   # ~9 GB download; default timeout fails
uv pip install "numpy<2" scipy scikit-learn pandas matplotlib seaborn tqdm pillow \
               matplotlib-venn venn==0.1.3 upsetplot==0.9.0 jupyter
uv pip install -e .                  # makes `import miae` work
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

`numpy<2` is a safety pin: the code was written against numpy 1.25.

### B3. Known issues to fix before running

These were found by reading the code. Fix them in your copy and note each one in your review notes (they are reproducibility findings).

| # | Where | Problem | Fix |
|---|---|---|---|
| 1 | every `experiment_scripts/*.sh`, `ensemble/ensemble_roc.py`, notebooks | `DATA_DIR="data"` placeholder | Set it to an absolute path, e.g. `~/Projects/mia_as_privacy/data` |
| 2 | `experiment_scripts/run_multi_seed.sh` | `CUDA_VISIBLE_DEVICES=1`. You have only GPU 0, so it would see no GPU at all | Change it to `0` |
| 3 | `run_multi_seed.sh` | Launches all seeds **in parallel** on one GPU, which will run out of memory on 8 GB | Run seeds one after another (loop `obtain_pred.sh $sd`), or at most 2 at a time |
| 4 | `miae/attacks/{calibration,aug,shokri}_mia.py` | `torch.load(...)` of whole models/datasets. PyTorch ≥ 2.6 defaults to `weights_only=True` and will refuse | `export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`, or add `weights_only=False` to those calls |
| 5 | `obtain_pred.py` argparse | `type=bool` flags: any non-empty string, **including `"False"`**, becomes `True` | Not a problem for the standard runs (only `--save_dataset`/`--train_target_model` matter, and both are passed as `"True"`). Note it, and don't rely on passing `"False"` |
| 6 | `datasets.py` | CIFAR is downloaded to `./data` **relative to the current directory** | Always run from `experiment/mia_comp/` (as the README says) |
| 7 | `obtain_venn.sh` | Uses `experiment_dir=".../mia_standard_exp"`, but predictions are written to `miae_standard_exp`. Output goes to `venn_lira_online_vs_offline/` | Fix the directory name and choose a sensible output folder |
| 8 | `obtain_jaccard.sh` | Reads from `repeat_miae_standard_exp/miae_standard_exp_${seed}/graphs/venn/...` and calls `../obtain_jaccard.py` | Point it at where `obtain_venn.sh` actually wrote, and fix the script path |
| 9 | all `*.sh` | The config is chosen by commenting lines in and out | Edit each one to `datasets=("cifar10")`, `archs=("resnet56")` before running |
| 10 | `prepare_target.sh` | Without `--dataset_file_root`, CINIC-10 can't load | Only matters if you do CINIC-10 (needs `process_CINIC10.ipynb` first) |

**Prepared changes (already applied in `code/`):**
- `prepare_target.sh` and `obtain_pred.sh` read their settings from environment variables (`DATA_DIR`, `DATASETS`, `ARCHS`, `MIAS`, `EXP_NAME`, `SHUFFLE_SEED`, `PREP_ROOT`, `DEVICE`, `EPOCHS`). The defaults are the paper's settings, so issues 1 and 9 no longer need editing. `DATA_DIR` defaults to `~/Projects/mia_as_privacy/data`.
- Both scripts export `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` (issue 4) and stop at the first error.
- `prepare_target.sh` passes `--dataset_file_root` (issue 10) and skips re-saving a dataset split that already exists.
- `obtain_pred.sh` uses one scratch folder per seed, dataset, architecture and attack (fix 11 in D6).
- `run_multi_seed.sh` runs seeds one after another on the default GPU (issues 2 and 3). `PARALLEL=1` restores the original behaviour.
- New: `experiment_scripts/smoke_test.sh` and `experiment_scripts/check_preds.py`.
- Bugs found while running the smoke test, fixed without changing results:
  - `obtain_pred.py`: the dataset-saving call from `prepare_target.sh` has no `--target_model`, so the model check crashed it. The published script hid this by carrying on after errors. The check is now skipped for `--save_dataset`.
  - `miae/utils/dataset_utils.py`: `torch._utils._accumulate` no longer exists in recent PyTorch. Replaced with `itertools.accumulate`, which gives the same split.
  - `miae/attacks/aug_mia.py`: `scipy.ndimage.shift` was called on all images at once. SciPy pads every axis and makes a float64 copy, so this needed ~17 GB for 15k images and ~34 GB for the 30k target images. That was out of memory on a 30 GB laptop, and the desktop crashed with it. The images are now shifted in chunks of 1,000. The output is identical (max abs difference 0.0 for all 9 shifts).
- Bugs found that change results (**not fixed**; report them in the review):
  - `aug_mia.py`, building the Augmentation training set: the class labels come from a `DataLoader` with `shuffle=True`, but the features come in dataset order. Labels end up paired with the wrong samples, so the per-class attack models train on randomly grouped data.
  - `aug_mia.py`, `create_translates`: the shifts `(0, dy, dx, 0)` assume images stored as (N, H, W, C), as in the TensorFlow original. Here they are (N, C, H, W), so the "translation" moves the color channels and the height, never the width.
- Still open: issues 5–8 (only matter for the analysis scripts) and D6 items 12–17.

Keep a `docs/REPRO_LOG.md` that records every change, the commands you ran, wall-clock times and any errors.

### B4. Stage 1: smoke test (about 1–2 h)

Goal: the pipeline runs end to end, and you get real timings. The scripts are already prepared (see "Prepared changes" in B3), so there is nothing to edit:

```bash
cd ~/Projects/mia_as_privacy/code/experiment/mia_comp
source ../../.venv/bin/activate
bash experiment_scripts/smoke_test.sh quick   # ~75 min: all 7 attacks at 2 epochs. Checks that the code runs; AUCs are ~0.5 by design
bash experiment_scripts/smoke_test.sh full    # ~1-2 h: paper settings, LOSS + calibration. Real timings and AUCs
bash experiment_scripts/smoke_test.sh lira    # optional: LiRA seed 0 at paper settings (many hours on the laptop)
```

Results go to `data/smoke_quick/` and `data/smoke_full/`, never to the main experiment folder. Each run ends with `check_preds.py`, which prints the target model's train/test accuracy and, for every attack, the shape, AUC, accuracy and TPR at 1% / 0.1% FPR. Wall-clock times per step are in `timings.log` in the same folder.

What to check after `full`:
- Train accuracy is high and test accuracy somewhat lower (a generalization gap exists).
- Calibration AUC is about 0.60 (Table 1, single instance). LOSS should be lower.
- Multiply the LiRA time by 6 seeds to plan the main run, or use it to correct the Part D estimates.

### B5. Stage 2: main run on CIFAR-10 / ResNet-56 (multi-day, background)

1. Set `mias=("lira" "reference" "shokri" "losstraj" "calibration" "yeom" "aug")`. Keep **lira before reference**, because reference reuses LiRA's shadow models from `lira_shadow_ckpts/`.
2. Run seeds 0–5 one after another, e.g.:
   ```bash
   for sd in 0 1 2 3 4 5; do bash experiment_scripts/obtain_pred.sh $sd > $DATA_DIR/output_$sd.log 2>&1; done
   ```
   Run this inside `tmux` so it survives a closed terminal. The script skips attacks whose `pred_*.npy` already exists, so it can be resumed safely.
3. After each seed, check that all 7 `pred_*.npy` files exist and that the AUCs are plausible.
4. Optional: seeds 6–15, for the 16-instance convergence curves (Figs 7–9).

### B6. Stage 3: reproduce figures and tables

Run everything from `experiment/mia_comp/`, with the scripts edited to cifar10/resnet56 and seeds 0–5.

| Paper item | What it shows | How to reproduce |
|---|---|---|
| **Fig. 4a** (ResNet-56 bars) | Consistency per attack < 0.4, except LOSS = 1 | `attack_consistency.ipynb`, or your own Part A4 code on the saved preds at FPR 0.1 |
| **Fig. 1b, Fig. 2** | Venn diagrams of 3 instances per attack | `obtain_venn.sh`, category `single_attack`, fpr 0.1 |
| **Fig. 1a** | Venn of LiRA vs. losstraj vs. reference | `obtain_venn.sh`, category `fpr`, subcategory `common_tp` |
| **Figs 7, 8, 9** | Coverage/stability TPR, FPR and precision vs. number of instances | `obtain_multi_seed_conv.sh` (set `seeds` to what you have) |
| **Fig. 10** | Pairwise Jaccard heatmaps of coverage and stability | `obtain_venn.sh` (pairwise) → `obtain_jaccard.sh` |
| **Fig. 11a/b** | Similarity vs. FPR | Same, with fprs 0.001 to 0.8 |
| **Fig. 12** | PCA of 𝒜-unique members | `disparity_empirical_analysis.ipynb` |
| **Fig. 13** | Top-x Class-NN | `obtain_pred_top_k_shokri.sh`, then `same_attack_different_signal.ipynb` (optional, extra GPU time) |
| **Fig. 15, 16, Table 1 (CIFAR-10 column)** | Ensemble ROC curves and AUC/ACC/TPR@0.1%FPR | `ensemble_roc.sh` with `datasets="cifar10" archs="resnet56"`; `max_ensemble_low_fpr.ipynb`; `ensemble_performance.ipynb` |
| **Fig. 17** | Cost vs. performance | `ensemble_cost_analysis.ipynb` (needs your timings from the logs) |
| **Figs 5, 6** | Disparity factors (fixed init, fixed attack set, number of shadow models) | Need code changes and extra runs. Optional, do last |

Cross-check at least consistency, coverage/stability and the stability ensemble with your **own** Part A4 implementation. Don't rely only on the authors' plotting code.

### B7. Numbers to compare against (CIFAR-10, ResNet-56)

Single instance (Table 1):

| Attack | AUC | ACC | TPR@0.1%FPR |
|---|---|---|---|
| losstraj | 0.635 | 0.588 | 0.002 |
| reference | 0.608 | 0.596 | 0.010 |
| lira | 0.585 | 0.570 | 0.005 |
| calibration | 0.603 | 0.572 | 0.005 |

Full multi-attack ensembles: Coverage 0.841 / 0.781 / 0.025, **Stability 0.863 / 0.770 / 0.050**, Majority vote 0.858 / 0.789 / 0.056.

Other targets:

- Consistency on CIFAR-10 is below about 0.4 for all attacks except LOSS (Fig. 4a).
- Coverage Jaccard is mostly below 0.4, and stability Jaccard mostly below 0.1. LiRA–reference is about 0.39 for coverage and about 0.16 for stability; losstraj–reference is about 0.63 for coverage (Fig. 10).
- Stability precision for losstraj, calibration and LiRA exceeds 95% with 16 instances (Fig. 9b).

Record your values next to these in `docs/RESULTS.md`, with the difference and your explanation of any gap.

---

## Part C: Points to probe in the review

Use your reproduction to support or refute each of these.

1. **Is the disparity beyond chance?** At FPR 0.1, with AUC about 0.6, TPR is low, so predicted sets are small and noisy. Compare the observed consistency and Jaccard values with the independent-detector baseline from A4. The paper gives no such null baseline.
2. **Oracle thresholds.** AdjustFPR picks thresholds using the target's true membership labels. That is fine for comparing attacks, but the ensemble's improvement relies on calibrating many instances perfectly. Does it hold if thresholds are set on held-out or shadow data instead?
3. **Compute-matched comparison.** The ensemble uses 6 instances × 4 attacks, versus 1 instance of 1 attack. Is it fairer to compare against LiRA with more shadow models (the original LiRA uses 64–256, the paper uses 20)? Fig. 6 touches on this only for consistency, not for TPR.
4. **Statistical uncertainty.** With 15k non-members, 0.1% FPR means about 15 false positives. Table 1 reports no confidence intervals or variance across target models. The paper mentions "4 experiment runs" only for Fig. 10.
5. **Weak-attack regime.** ResNet-56 is chosen for its small generalization gap, so all attacks are weak (AUC about 0.6). Do the conclusions hold for more overfit targets (CIFAR-100 shows higher consistency)?
6. **ROC of a set-based ensemble.** Check how `ensemble_roc.py` turns (base FPR → ensemble TPR/FPR) points into AUC (interpolation, the `--no_interp` flag). Are the reported AUC gains (e.g. "36%") sensitive to that choice?
7. **Reproducibility of the artifact.** Record everything from B3: outdated environment, hard-coded GPU index, path mismatches, the argparse bool bug, and which figures and tables could not be regenerated from the scripts alone.
8. **Explanatory analysis (Sec. 4.5).** It is explicitly "preliminary": PCA on 𝒜-unique samples, and Top-x Class-NN. Judge whether it supports the causal wording used.

---

## Part D: Full reproduction on sciCORE (SLURM)

Decision (supervisor): reproduce the **whole paper**. The only allowed reduction is the **number of shadow-model settings in Fig. 6**.

Cluster docs: https://docs.scicore.unibas.ch/HPC%20Cluster/batchcomputing/

### D1. sciCORE facts that drive the design

| Partition | GPUs | Per-user limit, QoS ≤ 6 h | Per-user limit, QoS 1 day | Use for |
|---|---|---|---|---|
| `l40s` | 2 nodes × 8 L40S (48 GB) = **16** | 16 GPUs | **4 GPUs** | The CIFAR-10/ResNet-56 main set, so timings match the paper's L40S numbers (Fig. 17) |
| `a100` (+ `a100-80g`) | 11 × 4 A100-40GB (+ 3 × 4 A100-80GB) | 48 GPUs | **24 GPUs** | **Everything else** |
| `h200`, `rtx4090`, `titan` | various | | | Not needed |

- These limits apply **per group as well as per user**. Lab mates using the same GPUs reduce your share.
- Maximum walltime: 6 h (`*-6hours`), 24 h (`*-1day`), 7 days (`*-1week`, with fewer GPUs allowed). The default is 1 CPU and 1 GB RAM, so always request resources explicitly.
- `$TMPDIR` is per-job local disk, deleted when the job ends (sciCORE rejects `--tmp=`, so its size cannot be requested). Home is about 1 TB per group volume and backed up. Name heavy result folders `*_NOBACKUP`.
- The directory for `--output` logs must exist before you submit, or the job fails silently.

### D2. Cost model

All figures are in **L40S GPU-hours**, based on the paper's appendix C.1 (CIFAR-10, ResNet-56, one seed):

- **LiRA:** 580 min, i.e. ~27 min per shadow model × 20, plus ~40 min of queries.
- **Reference:** reuses LiRA's shadow models; its queries take about 30 min (my estimate).
- **"Fast bundle"** (losstraj 17 + calibration 5 + Class-NN ~30 + Aug ~30 + LOSS ~5 min): ≈ 1.5 h. Only the losstraj and calibration figures are from the paper; the rest are my estimates.
- **One seed of all 7 attacks ≈ 11.7 h.**

Scaling assumptions, to be **checked in the pilot (D5)**:

| Factor | Value | Reason |
|---|---|---|
| CIFAR-10, CINIC-10 | ×1.0 | Same sizes (15k/15k/30k), 60 epochs |
| CIFAR-100 | ×1.67 | 100 epochs instead of 60 |
| ResNet-56 / VGG-16 | ×1.0 | Small or GPU-efficient models; jobs appear limited by data loading |
| MobileNetV2 | ×1.25 | |
| WideResNet-32-4 | ×2.0 | ~8× the FLOPs of ResNet-56 |
| Purchase-100 / Texas-100 (MLP) | ~0.5 h per seed for all attacks | Tiny MLP, LiRA with 1 augmentation |

### D3. Complete job inventory

A **seed set** means all 7 attacks for one (data partition, dataset, architecture, seed). Each seed set is split into **3 jobs**:

1. **LiRA:** trains the 20 shared shadow models, then runs inference.
2. **Reference:** reuses LiRA's shadow models, so it must start *after* job 1.
3. **Fast bundle:** losstraj, calibration, Class-NN (`shokri`), aug and LOSS (`yeom`), run one after another.

| ID | Experiment | Paper items | Grid | Seed sets | Jobs | GPU-h |
|---|---|---|---|---|---|---|
| S | Data splits + target models | prerequisite; Table 2 | 1 split job per (partition, dataset); 1 target model per (partition, dataset, arch) | | ~12 CPU + ~22 GPU | ~15 |
| A | **Main grid** | Fig. 1, 2, 4, 10–12, 15–17; Tables 1, 7 | {CIFAR-10, CIFAR-100, CINIC-10} × {ResNet-56, WRN-32-4, VGG-16, MobileNetV2} × seeds 0–5 | 72 | 216 | **~1,350** |
| B | Extra seeds for convergence | Figs 7–9 (16 instances) | {CIFAR-10, CIFAR-100} × ResNet-56 × seeds 6–15 | 20 | 60 | ~310 |
| C | Repeated data partitions | Fig. 10 "average of 4 runs"; Table 2 | 3 extra partitions (`shuffle_seed` 2, 3, 4) × {CIFAR-10, CIFAR-100} × ResNet-56 × seeds 0–5 | 36 | 108 | ~560 |
| D | Tabular datasets | appendix | {Purchase-100, Texas-100} × MLP × 6 attacks (no aug) × seeds 0–5 | 12 | 12 | ~6 |
| E | Disparity factors | Fig. 5a (same initial weights), 5b (same attack training set) | CIFAR-10 × ResNet-56 × 6 seeds; 5b reuses shadows and retrains only the attack models | 6 + 6 | 24 | ~75 |
| F | **Shadow-model count (reduced)** | Fig. 6 | Calibration with **1, 3, 7, 13** shadows (1 reuses A); LiRA + Reference with **20, 60, 100** (20 reuses A); CIFAR-10 × ResNet-56 × 6 seeds | | 18 + 24 | **~500** |
| G | Top-x Class-NN | Fig. 13 | top-1/3/10 × CIFAR-10 × ResNet-56 × 6 seeds | | 18 | ~9 |
| H | Offline LiRA | appendix (consistency) | CIFAR-10 × ResNet-56 × 6 seeds, reusing A's shadow models | | 6 | ~6 |
| I | Auxiliary/target distribution gap | appendix | Class-NN + LOSS, CIFAR-10 ↔ CINIC-10, ResNet-56, seeds 0–2 | | ~12 | ~5 |
| | **Total** | | | | **~520 GPU jobs** | **≈ 2,850** |

With a 15% margin for failed runs and re-runs, request **≈ 3,300 GPU-hours**. Each job also uses 10 CPU cores, so that is about **33,000 CPU core-hours**, in case sciCORE accounts for CPU time separately.

**Fig. 6 options** (the only allowed reduction):

| LiRA/Reference shadow counts | Calibration counts | Extra GPU-h |
|---|---|---|
| 20, 40, 60, 80, 100 (as in the paper) | 1, 3, 5, …, 13 | ~850 |
| **20, 60, 100 (recommended)** | **1, 3, 7, 13** | **~500** |
| 20, 40, 60 | 1, 3, 7, 13 | ~300 (loses the 80–100 tail) |

LiRA shadow models can't be shared between counts. The code's data split depends on the total number of shadow models (`_split_data(dataset, expid, num_shadow_models, seed)`), so each count needs its own set.

### D4. Per-job resources and walltimes

All jobs request `--gres=gpu:1 --cpus-per-task=10 --mem=32G`. Walltime is about 1.5× the estimate:

| Job type | ResNet-56 / VGG-16, CIFAR-10 or CINIC-10 | Worst case (WRN-32-4, CIFAR-100) | QoS |
|---|---|---|---|
| LiRA (20 shadows) | est. 10 h → `--time=16:00:00` | est. 32 h → **2 submissions** of 24 h (it resumes) | `a100-1day` / `l40s-1day` |
| Reference | est. 0.5 h → `--time=02:00:00` | est. 1.5 h → `04:00:00` | `*-6hours` |
| Fast bundle | est. 1.5 h → `--time=03:00:00` | est. 5 h → `--time=08:00:00` | `*-6hours`, or `*-1day` if over 6 h |
| Target model | est. 0.5 h → `02:00:00` | est. 1.5 h → `04:00:00` | `*-6hours` |
| Fig. 6 LiRA, 100 shadows | est. 45 h → 2–3 submissions of 24 h (resumes) | | `a100-1day` |
| Tabular (all attacks) | est. 0.5 h → `02:00:00` | | `a100-6hours` |
| Analysis (Venn, Jaccard, ensemble ROC) | CPU only, `scicore` partition, 8 CPUs, 32 GB, ≤ 6 h | | `6hours` |

**Wall-clock time:**

- On A100s, 24 GPUs at once (`a100-1day` per-user limit): 2,850 / 24 ≈ 120 h, so about **5 days if never queued**. Plan for **2–4 weeks** with real queue waits.
- Only the 6-seed CIFAR-10/ResNet-56 set (~70 GPU-h) goes on `l40s`: 4 at once, about 1 day.
- Short jobs (≤ 6 h) may use more GPUs at once under the `*-6hours` QoS (48 on A100).

### D5. Before submitting everything: pilot (~25 GPU-h)

1. CIFAR-10/ResNet-56, seed 0: run all 3 jobs on **`l40s`** and on **`a100`**. Compare the times with D2. This also checks the L40S → A100 speed ratio.
2. CIFAR-100 + WRN-32-4, seed 0, LiRA only, for 2 h. Extrapolate from the per-shadow time in the log. This checks the worst-case multiplier.
3. Run `seff <jobid>` on each pilot job to see real CPU, RAM and GPU use, then adjust `--cpus-per-task` and `--mem`.
4. Update the D3 totals with the measured numbers *before* submitting the rest.

**How to run it** (scripts in `code/experiment/mia_comp/slurm/`):

```bash
# from your PC (UniBas network or VPN): login with your UniBas account name, not your e-mail
ssh <user>@login12.scicore.unibas.ch
git clone https://github.com/Den55000/mia-as-privacy.git && cd mia-as-privacy/code/experiment/mia_comp
bash slurm/setup_env.sh                       # once, on the login node: venv (torch 2.0.1+cu118) + CIFAR download
sacctmgr show qos format=name%20,maxwall,maxtrespu%30 | grep -E 'l40s|a100'   # check the QoS names exist
sbatch slurm/gpu_check.sbatch                 # 5 min, A100; then the same with --partition=l40s --qos=l40s-30min
bash slurm/submit_pilot.sh                    # 9 jobs, ~30 GPU-h; results in ~/mia_data_NOBACKUP/pilot_*
squeue --me                                   # while it runs
bash slurm/pilot_report.sh                    # afterwards: times, seff, GPU use, per-shadow time, AUCs
```

The WRN probe (`mia_wrnprobe`) is *meant* to hit its 6 h limit, so it shows as `TIMEOUT`. The report extrapolates the time for 20 shadow models from the checkpoints it finished.

### D6. Code changes needed for a cluster run and the full paper

In addition to B3:

| # | Change | Needed for |
|---|---|---|
| 11 | Pass `--preparation_path $TMPDIR/prep` for every job. `obtain_pred.sh` uses one `prepare_sd{seed}` folder per seed and **deletes it at the start**, so parallel jobs for the same seed would destroy each other's files | all |
| 12 | Before resubmitting a timed-out LiRA job, delete the **last, incomplete** shadow folder (it has no `shadow.pth`). Otherwise `len(os.listdir(shadow_path)) >= num_shadow_models` causes **all** training to be skipped, leaving a missing model | resumed LiRA jobs |
| 13 | Add a `--num_shadow_models` CLI flag. It is hardcoded to 20 for LiRA/Reference and 1 for calibration in `get_aux_info()` | Fig. 6 |
| 14 | Add a flag for `shadow_diff_init` (hardcoded `True`), and a mode that reuses fixed shadow models / attack training set | Fig. 5 (no code for it in the repo) |
| 15 | Add a `top_k` CLI flag (hardcoded to 10 for `top_k_shokri`) | Fig. 13 |
| 16 | `obtain_accuracy.py` is called by `obtain_accuracy.sh` but **missing from the repo** | accuracy tables |
| 17 | The distribution-gap experiment setup is not documented in the scripts | appendix |

Items 14, 16 and 17 mean some results depend on **guessing the authors' setup**. Consider emailing the first author (wangz56@rpi.edu) for the missing pieces. Either way, record it as a reproducibility finding.

### D7. Submission layout

- Write one text file of commands per job type (`jobs/lira.txt`, `jobs/reference.txt`, `jobs/fast.txt`, …), one line per task. Submit each as a job array, as in the sciCORE array example.
- Chain them with dependencies:
  1. splits
  2. → target models
  3. → LiRA array + fast array
  4. → reference array, with `--dependency=aftercorr:<lira_array_id>` so task *i* waits for LiRA task *i*
  5. → CPU analysis jobs

```bash
#!/bin/bash
#SBATCH --job-name=mia_lira
#SBATCH --partition=a100
#SBATCH --qos=a100-1day
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --array=1-216%24
#SBATCH --output=logs/%x_%A_%a.out      # logs/ must exist before sbatch

source "$HOME/mia_as_privacy/code/.venv/bin/activate"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
cd "$HOME/mia_as_privacy/code/experiment/mia_comp"
cmd=$(sed -n "${SLURM_ARRAY_TASK_ID}p" jobs/lira.txt)   # each line: python3 obtain_pred.py ... --preparation_path "$TMPDIR/prep"
echo "$cmd"; eval "$cmd"
```

- Check what failed with `sacct -j <id> --format=JobID,State,Elapsed,MaxRSS`. Resubmit only `TIMEOUT` or `FAILED` tasks with `--array=<list>`, after applying fix 12.

## Suggested schedule

(This schedule is for the laptop / Tier 1 path. For the full paper, use Part D: setup and pilot in week 1, bulk runs in weeks 2–4, analysis in weeks 4–5.)

| Day | Work |
|---|---|
| 1–2 | Part A1–A3 reading, A5 questions |
| 2 | A4 toy notebook; B1–B3 setup and code fixes |
| 3 | B4 smoke test; start B5 seeds in the background |
| 3–6 | While the GPU runs: read the code in depth (`miae/attacks`, `prediction.py`, `ensemble_roc.py`), and draft Part C notes |
| 6–8 | B6 figures, B7 comparison, write `RESULTS.md` |
| 8+ | Optional extras (16 seeds, CIFAR-100, Figs 5/6/13), then write the review |

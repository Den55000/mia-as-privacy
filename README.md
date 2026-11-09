# Reviewing and reproducing "Membership Inference Attacks as Privacy Tools"

This is a student project at the University of Basel. I review and reproduce:

> Zhiqi Wang, Chengyu Zhang, Yuetian Chen, Nathalie Baracaldo, Swanand R. Kadhe, Lei Yu.
> **Membership Inference Attacks as Privacy Tools: Reliability, Disparity and Ensemble.**
> ACM CCS 2025. https://doi.org/10.1145/3719027.3744818 · extended version: [arXiv:2506.13972](https://arxiv.org/abs/2506.13972)
> Original code: https://github.com/RPI-DSPlab/mia-disparity

**This is not the authors' repository.** The paper, its ideas and the attack implementations are the work of the authors above. This repository has my review report, my reproduction plan and log, and the small changes I needed to run their code.

## What the paper shows, in short

Membership inference attacks (MIAs) guess whether a data point was in a model's training set, and they are widely used to measure privacy risk. The paper shows two things:

- Running the **same attack** with a different random seed flags largely different members.
- **Different attacks** at the same false-positive rate also flag different members.

So a single run of a single attack underestimates the risk. The authors propose ensembles of attacks instead.

## What is in this repository

| Path | Contents |
|---|---|
| `report/` | My review report (LaTeX) and its supplement on the use of AI tools (`appendix.tex`/`.pdf`). The figures in `report/figures/wang2025-figs.pdf` are taken from the paper and cited there |
| `docs/PLAN.md` | Plan for understanding and reproducing the paper, including the compute budget for the full reproduction on the sciCORE cluster |
| `docs/REPRO_LOG.md` | Log of every change to the authors' code, the commands run, run times and errors |
| `code/` | The authors' repository at commit [`71a2965`](https://github.com/RPI-DSPlab/mia-disparity/tree/71a2965), imported with `git subtree`, plus my changes |
| `code/experiment/mia_comp/slurm/` | My job scripts for running the pilot on sciCORE (SLURM) |

### My changes to the authors' code

The changes are listed in `docs/REPRO_LOG.md` and marked in the source with an `[AI-generated: ...]` comment. In short:

- I made the run scripts configurable and stopped them from overwriting each other's files.
- I fixed one compatibility problem with newer PyTorch and one memory problem in the Augmentation attack. Neither fix changes the results.
- I found two bugs in the Augmentation attack that **do** change its results. I left them unfixed so the paper's numbers can be reproduced as published, and report them in the review.

To see every change, compare `code/` with the upstream repository at commit `71a2965`.

## Status

- Done: a smoke test (all 7 attacks, CIFAR-10/ResNet-56, 2 epochs) ran on a laptop GPU.
- Next: a pilot on sciCORE (`docs/PLAN.md`, section D5), then the full reproduction.

## Licenses

- `code/` is the authors' code under the MIT License (`code/LICENSE.md`, © 2025 RPI-DSPLab). My changes in `code/` are under the same license.
- I did not put the paper's PDF in this repository, because the ACM copyright notice does not allow reposting it. Use the DOI or arXiv links above.

## Use of AI tools

I wrote the code changes, the plan and drafts of the report with the help of Claude Code (Anthropic). The report's supplement (`report/appendix.tex`) lists the tools and every prompt I gave.

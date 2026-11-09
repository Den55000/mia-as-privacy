#!/bin/bash
# [AI-generated: Claude Code, Claude Opus 5.5, 2026-11-09] Entire file.
# One-time setup on the sciCORE LOGIN node (login12), which has internet; compute nodes may not.
# Run from experiment/mia_comp/:  bash slurm/setup_env.sh
#
# Builds code/.venv with the repo's ORIGINAL versions (PyTorch 2.0.1 + CUDA 11.8, see miae_env.yml),
# which run on the cluster's L40S/A100 GPUs (PLAN.md, D/cluster notes), and pre-downloads
# CIFAR-10/100 into ./data, where datasets.py looks for them.
set -euo pipefail

code_dir="$(realpath ../..)"

if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

cd "$code_dir"
uv venv --python 3.11 .venv
source .venv/bin/activate
# torch from the PyTorch index only, so uv cannot pick the PyPI build of the same version
UV_HTTP_TIMEOUT=600 uv pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
# remaining versions as pinned in miae_env.yml
uv pip install numpy==1.25.0 scipy==1.10.1 scikit-learn==1.2.2 pandas==1.5.3 matplotlib==3.7.1 \
               seaborn==0.12.2 tqdm==4.65.0 pillow==9.4.0 matplotlib-venn==0.11.10 \
               venn==0.1.3 upsetplot==0.9.0
uv pip install -e .

cd "$code_dir/experiment/mia_comp"
python - <<'EOF'
import torch, torchvision
print("torch", torch.__version__, "CUDA build", torch.version.cuda)
torchvision.datasets.CIFAR10(root="./data", train=True, download=True)
torchvision.datasets.CIFAR10(root="./data", train=False, download=True)
torchvision.datasets.CIFAR100(root="./data", train=True, download=True)
torchvision.datasets.CIFAR100(root="./data", train=False, download=True)
import miae; print("miae import OK")
EOF
mkdir -p logs
echo "Done. Next: sbatch slurm/gpu_check.sbatch"

# [AI-generated: Claude Code, Claude Opus 5.5, 2026-11-10] Entire file.
"""
Download CINIC-10 and draw the paper's 60k subset, as cells 1-3 of the authors'
process_CINIC10.ipynb do, as a script for the sciCORE login node (compute nodes have no
internet). Same source URL, seed (set_seed(0)), folders and 3,000 random images per class.

One deviation: the class and image listings are sorted before sampling. The notebook samples
from os.listdir() order, which depends on the file system, so its subset cannot be
reproduced exactly anywhere; sorting at least makes ours reproducible (REPRO_LOG.md).
Only the train/ and test/ folders are extracted (the notebook uses no others).

Usage (from experiment/mia_comp/, on the login node):
    python experiment_scripts/prepare_cinic10.py [out_dir]     # default ~/mia_data_NOBACKUP/CINIC10
Result: <out_dir>/CINIC10_60ksubset/{cifar10_train_subset,imagenet_test_subset}/<class>/*.png
"""
import os
import random
import shutil
import sys
import tarfile
import urllib.request

sys.path.append(os.path.join(os.getcwd(), "..", ".."))
from miae.utils.set_seed import set_seed  # noqa: E402

URL = "https://datashare.is.ed.ac.uk/bitstream/handle/10283/3192/CINIC-10.tar.gz?sequence=4&isAllowed=y"
PER_CLASS = 3000


def extract_subset(src, n, dst):
    if os.path.exists(dst):
        shutil.rmtree(dst)
    for c in sorted(os.listdir(src)):
        os.makedirs(os.path.join(dst, c))
        for img in random.sample(sorted(os.listdir(os.path.join(src, c))), n):
            shutil.copy(os.path.join(src, c, img), os.path.join(dst, c))
        print(f"  {dst}/{c}: {n} images")


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/mia_data_NOBACKUP/CINIC10")
    os.makedirs(out, exist_ok=True)
    set_seed(0)  # as in the notebook; seeds Python's random, which random.sample uses

    tar_path = os.path.join(out, "CINIC-10.tar.gz")
    if not os.path.exists(tar_path):
        print(f"Downloading {URL} ...")
        urllib.request.urlretrieve(URL, tar_path + ".part")
        os.rename(tar_path + ".part", tar_path)
    if not (os.path.isdir(os.path.join(out, "train")) and os.path.isdir(os.path.join(out, "test"))):
        print("Extracting train/ and test/ ...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(out, members=[m for m in tar if m.name.split("/")[0] in ("train", "test")])

    sub = os.path.join(out, "CINIC10_60ksubset")
    extract_subset(os.path.join(out, "train"), PER_CLASS, os.path.join(sub, "cifar10_train_subset"))
    extract_subset(os.path.join(out, "test"), PER_CLASS, os.path.join(sub, "imagenet_test_subset"))
    # the full extraction (180k files) is no longer needed; the archive is kept
    shutil.rmtree(os.path.join(out, "train"))
    shutil.rmtree(os.path.join(out, "test"))
    print(f"Done: {sub}")


if __name__ == "__main__":
    main()

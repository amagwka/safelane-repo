"""
Write the ten offline datasets used in the article to data/ as .npz files.
Each file holds the raw logged transitions (s, a, r, c, s2, d) for one
dataset seed, exactly as run_experiment.py generates them.
"""

import pathlib
import sys

import numpy as np

sys.path.insert(0, ".")
import algos as AL                       # noqa: E402
from run_experiment import SEEDS, N_TRAIN_EP, BEH_EPS  # noqa: E402

out = pathlib.Path("data")
out.mkdir(exist_ok=True)
for seed in SEEDS:
    d = AL.collect_dataset(seed=seed, n_episodes=N_TRAIN_EP, eps=BEH_EPS)
    np.savez_compressed(out / f"dataset_seed{seed:02d}.npz", **d)
    print(f"seed {seed:2d}: {len(d['a']):6d} transitions")

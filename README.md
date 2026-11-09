# SafeLane-v0 — conservative offline RL for safety-constrained lane-change control

Code and data for the article *Conservative Offline Reinforcement Learning for
Safety-Constrained Lane-Change Control: A Pilot Study*
(Kembayev Mukhamejan, Astana IT University, Research Methods and Tools).

## Contents
| path | purpose |
| --- | --- |
| `env.py` | the SafeLane-v0 simulator (constrained MDP: reward and safety cost are separate signals) |
| `algos.py` | offline dataset collection, the five compared policies, the evaluator |
| `run_experiment.py` | runs all seeds, the ablation and the noise sweep; writes `results.json` |
| `plots.py` | renders the two figures as vector PDFs into `figs/` |
| `make_tables.py` | emits the LaTeX tables from `results.json` |
| `export_datasets.py` | writes the ten offline datasets to `data/` |
| `data/dataset_seedXX.npz` | the ten offline datasets (arrays `s, a, r, c, s2, d`, about 39,000 transitions each) |
| `results.json` | every number reported in the article |
| `figs/` | the two figures used in the article |

## Reproducing the results
```bash
pip install -r requirements.txt
python run_experiment.py     # ~5 min on 2 CPU cores, writes results.json
python plots.py              # figs/*.pdf
python make_tables.py        # tables/*.tex
python export_datasets.py    # data/*.npz
```
Datasets use seeds 1–10; seed 0 is reserved for hyperparameter tuning.
Evaluation episodes use environment seeds `50000 + 101*s` for dataset seed `s`.
Every number in the article comes from `results.json` without manual transcription.
Tested with Python 3.13; no GPU is needed.

## Licence
Code: MIT (see `LICENSE`). Generated datasets in `data/` and `results.json`: CC BY 4.0.

"""
Emit the LaTeX table bodies from results.json so that no number in the
paper is transcribed by hand. Writes tables/*.tex and prints a digest.
"""

import json
import pathlib

import numpy as np

R = json.loads(pathlib.Path("results.json").read_text())
T = pathlib.Path("tables"); T.mkdir(exist_ok=True)

LABEL = {
    "behaviour": "Behaviour policy (data logger)",
    "bc": "Behavioural cloning",
    "naive": "Naive offline VI ($\\alpha=0$)",
    "covi": "Conservative offline VI ($\\alpha^{*}=2$)",
    "covi_shield": "Conservative VI $+$ cost shield",
}
ORDER = ["behaviour", "bc", "naive", "covi", "covi_shield"]


def ms(d, p=2):
    return f"{d['mean']:.{p}f} $\\pm$ {d['std']:.{p}f}"


def ci(d, p=2):
    lo, hi = d["ci"]
    return f"[{lo:.{p}f}, {hi:.{p}f}]"


# ------------------------------------------------- Table I: the dataset
ds = R["dataset"]
raw = np.array([d["raw_transitions"] for d in ds])
cln = np.array([d["clean_transitions"] for d in ds])
rem = np.array([d["removed"] for d in ds])
cov = 100 * np.array([d["sa_coverage"] for d in ds])
vis = np.array([d["states_visited"] for d in ds])
uns = 100 * np.array([d["unsafe_fraction"] for d in ds])
acts = np.array([d["action_hist"] for d in ds]).mean(axis=0) * 100
trunc = np.array([d["reasons"]["horizon-truncated last step"] for d in ds])
nonfin = np.array([d["reasons"]["non-finite reward or cost"] for d in ds])
cfg = R["config"]

rows = [
    ("Source / logging policy",
     f"SafeLane-v0 simulator; hand-coded controller, $\\varepsilon$-greedy, $\\varepsilon={cfg['behaviour_eps']}$"),
    ("Datasets (seeds) $\\times$ episodes",
     f"{len(ds)} (seeds {min(cfg['seeds'])}--{max(cfg['seeds'])}) $\\times$ {cfg['n_train_episodes']}; seed 0 for tuning only"),
    ("Transitions: raw $\\rightarrow$ clean",
     f"{raw.mean():.0f} $\\rightarrow$ {cln.mean():.0f} $\\pm$ {cln.std(ddof=1):.0f} "
     f"({100*rem.mean()/raw.mean():.2f}\\% removed)"),
    ("Records removed, and why",
     f"{trunc.mean():.0f} horizon-truncated last steps (no valid bootstrap target); "
     f"{nonfin.mean():.0f} non-finite"),
    ("Train / val / test split",
     "all clean transitions are training data; no transition-level split, "
     "as policies are scored online"),
    ("State--action coverage",
     f"{cov.mean():.1f}\\% $\\pm$ {cov.std(ddof=1):.1f}\\% of {cfg['n_states']}$\\times${cfg['n_actions']} pairs; "
     f"{vis.mean():.0f} of {cfg['n_states']} states visited"),
    ("Target distribution",
     f"{uns.mean():.2f}\\% $\\pm$ {uns.std(ddof=1):.2f}\\% unsafe steps; actions "
     + "/".join(f"{a:.0f}" for a in acts) + "\\% (keep/left/right/acc/dec)"),
    ("Evaluation",
     f"{cfg['n_test_episodes']} episodes per seed and policy, held-out environment "
     "seeds, identical across policies"),
]
with open(T / "tab_dataset.tex", "w") as f:
    for k, v in rows:
        f.write(f"{k} & {v} \\\\\n")

# ------------------------------------------------- Table II: main results
with open(T / "tab_main.tex", "w") as f:
    for m in ORDER:
        d = R["main"][m]
        f.write(f"{LABEL[m]} & {ms(d['return'],1)} & {ci(d['return'],1)} & "
                f"{ms(d['unsafe_per100'])} & {ms(d['crash_rate'],3)} & "
                f"{d['ep_len']['mean']:.0f} \\\\\n")

# ------------------------------------------------- Table III: tests
def _holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    m, adj, run = len(p), np.empty(len(p)), 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj


# Re-apply Holm across each whole research-question family, as the Methods
# section states ("adjusted within each research question"). The runner
# corrected within each metric, which is a smaller family.
for _rq in ("RQ1", "RQ2"):
    _keys = [k for k in R["tests"] if k.startswith(_rq + "_")]
    _adj = _holm([R["tests"][k]["w_p"] for k in _keys])
    for _k, _a in zip(_keys, _adj):
        R["tests"][_k]["w_p_holm"] = float(_a)

NICE = {
    "covi_vs_behaviour": "Conservative vs.\\ behaviour",
    "covi_vs_bc": "Conservative vs.\\ cloning",
    "covi_vs_naive": "Conservative vs.\\ naive",
    "covi_shield_vs_covi": "Shield vs.\\ no shield",
}
MET = {"return": "Return", "crash_rate": "Collision rate",
       "unsafe_per100": "Unsafe/100 steps"}

lines = []
for key, t in R["tests"].items():
    rq, rest = key.split("_", 1)
    for met in ("unsafe_per100", "crash_rate", "return"):
        if rest.startswith(met + "_"):
            comp = rest[len(met) + 1:]
            break
    lo, hi = t["diff_ci"]
    sig = "yes" if t["w_p_holm"] < 0.05 else "no"
    lines.append((rq, f"{NICE[comp]} & {MET[met]} & {t['mean_diff']:+.3f} & "
                      f"[{lo:+.3f}, {hi:+.3f}] & {t['cohen_dz']:+.2f} & "
                      f"{t['w_p']:.4f} & {t['w_p_holm']:.4f} & {sig} \\\\"))
with open(T / "tab_tests.tex", "w") as f:
    for rq in ("RQ1", "RQ2"):
        f.write(f"\\multicolumn{{8}}{{l}}{{\\emph{{{rq}}}}} \\\\\n")
        for r, line in lines:
            if r == rq:
                f.write(line + "\n")

# ------------------------------------------------- digest for the prose
print(f"runtime {R['runtime_sec']}s | coverage {cov.mean():.1f}% | "
      f"clean {cln.mean():.0f} | shield changed states "
      f"{np.mean(R['shield_changes']):.1f}")
print("\n%-34s %8s %8s %8s %6s" % ("method", "return", "uns/100", "crash", "len"))
for m in ORDER:
    d = R["main"][m]
    print("%-34s %8.1f %8.2f %8.3f %6.0f" % (
        m, d["return"]["mean"], d["unsafe_per100"]["mean"],
        d["crash_rate"]["mean"], d["ep_len"]["mean"]))
print("\ntests:")
for k, t in R["tests"].items():
    print("  %-42s diff %+8.3f  p=%.4f  holm=%.4f  dz=%+.2f" % (
        k, t["mean_diff"], t["w_p"], t["w_p_holm"], t["cohen_dz"]))
print("\nablation (alpha: return / crash):")
for a in cfg["alpha_grid"]:
    d = R["ablation"][str(float(a))]
    print("  a=%-5g ret %7.2f +- %5.2f   crash %.3f +- %.3f" % (
        a, d["return"]["mean"], d["return"]["std"],
        d["crash_rate"]["mean"], d["crash_rate"]["std"]))
print("\nrobustness (sigma: return / crash):")
for m in ["behaviour", "bc", "naive", "covi"]:
    s = "  %-10s" % m
    for sg in cfg["noise_grid"]:
        d = R["robustness"][m][str(float(sg))]
        s += " s=%.1f:%6.1f/%.2f" % (sg, d["return"]["mean"], d["crash_rate"]["mean"])
    print(s)
print("\nevents per 100 episodes:")
n_ep = len(cfg["seeds"]) * cfg["n_test_episodes"]
for m in ORDER:
    print("  %-14s" % m, {k: round(100*v/n_ep, 2) for k, v in R["events"][m].items()})
print("\nunsafe rate by headway bucket (%):", R["gap_bucket_labels"])
for m in ORDER:
    print("  %-14s" % m, [round(100*x, 2) for x in R["gap_unsafe_rate"][m]])

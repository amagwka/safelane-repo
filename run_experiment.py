"""
Full pilot experiment. Writes results.json with every number used in the
Results section. Hyperparameters alpha* and (beta*, tau*) were frozen on a
separate tuning dataset (seed 0) before this script was run; the evaluation
seeds below were never used for tuning.
"""

import json
import sys
import time

import numpy as np
from scipy import stats

sys.path.insert(0, ".")
import env as E          # noqa: E402
import algos as AL       # noqa: E402

# ------------------------------------------------------------------ config
SEEDS = list(range(1, 11))          # 10 dataset seeds (seed 0 was the tuning set)
N_TRAIN_EP = 300                    # logging episodes per dataset
BEH_EPS = 0.20
N_TEST_EP = 200                     # online evaluation episodes, main table
N_TEST_EP_SWEEP = 100               # for the alpha ablation
N_TEST_EP_ROB = 80                  # for the noise sweep
ROB_SEEDS = SEEDS[:8]
TEST_SEED_BASE = 50_000

ALPHA_STAR = 2.0
BETA_STAR, TAU_STAR = 0.1, 0.25
ALPHA_GRID = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
NOISE_GRID = [0.0, 0.1, 0.2, 0.3, 0.4]
N_BOOT = 10_000


def summarise(ev):
    """Per-episode arrays -> the three metrics defined in the Methods."""
    return {
        "return": float(ev["return"].mean()),
        "unsafe_per100": float(100.0 * ev["cost"].sum() / ev["len"].sum()),
        "crash_rate": float(ev["crash"].mean()),
        "ep_len": float(ev["len"].mean()),
    }


def boot_ci(x, n_boot=N_BOOT, seed=12345):
    """Percentile bootstrap 95% CI of the mean over seeds."""
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_test(a, b):
    """Paired comparison across seeds: Wilcoxon (primary) + paired t-test."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    out = {
        "mean_diff": float(d.mean()),
        "diff_ci": boot_ci(d),
        "t_p": float(stats.ttest_rel(a, b).pvalue),
        "cohen_dz": float(d.mean() / d.std(ddof=1)) if d.std(ddof=1) > 0 else 0.0,
    }
    if np.allclose(d, 0):
        out["w_p"] = 1.0
    else:
        out["w_p"] = float(stats.wilcoxon(a, b).pvalue)
    return out


def holm(pvals):
    """Holm-Bonferroni adjusted p-values, order preserved."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * p[i]
        running = max(running, val)
        adj[i] = min(1.0, running)
    return adj.tolist()


# ------------------------------------------------------------------ main
def main():
    t0 = time.time()
    res = {
        "config": {
            "seeds": SEEDS, "n_train_episodes": N_TRAIN_EP,
            "behaviour_eps": BEH_EPS, "n_test_episodes": N_TEST_EP,
            "alpha_star": ALPHA_STAR, "beta_star": BETA_STAR, "tau_star": TAU_STAR,
            "gamma": AL.GAMMA, "gamma_cost": AL.GAMMA_C, "vi_iters": AL.VI_ITERS,
            "n_states": E.N_STATES, "n_actions": E.N_ACTIONS,
            "horizon": E.HORIZON, "dt": E.DT, "road_len": E.ROAD_LEN,
            "n_traffic": E.N_TRAFFIC, "ttc_unsafe": E.TTC_UNSAFE,
            "alpha_grid": ALPHA_GRID, "noise_grid": NOISE_GRID,
            "n_bootstrap": N_BOOT, "numpy": np.__version__,
        },
        "dataset": [], "main": {}, "ablation": {}, "robustness": {},
        "errors": {}, "coverage": [],
    }

    methods = ["behaviour", "bc", "naive", "covi", "covi_shield"]
    per_seed = {m: {k: [] for k in ("return", "unsafe_per100", "crash_rate", "ep_len")}
                for m in methods}
    ablation = {a: {k: [] for k in ("return", "unsafe_per100", "crash_rate")}
                for a in ALPHA_GRID}
    event_tot = {m: {k: 0 for k in AL.EVENTS} for m in methods}
    gap_rate = {m: [] for m in methods}
    gap_occ = {m: [] for m in methods}

    for si, seed in enumerate(SEEDS):
        raw = AL.collect_dataset(seed=seed, n_episodes=N_TRAIN_EP, eps=BEH_EPS)
        clean, n_removed, reasons = AL.filter_dataset(raw)
        n, rhat, chat, P, seen = AL.empirical_model(clean)

        res["dataset"].append({
            "seed": seed,
            "raw_transitions": int(len(raw["s"])),
            "clean_transitions": int(len(clean["s"])),
            "removed": n_removed, "reasons": reasons,
            "episodes": N_TRAIN_EP,
            "sa_coverage": float(seen.sum() / seen.size),
            "states_visited": int((n.sum(axis=1) > 0).sum()),
            "mean_reward": float(clean["r"].mean()),
            "unsafe_fraction": float(clean["c"].mean()),
            "action_hist": (np.bincount(clean["a"], minlength=E.N_ACTIONS)
                            / len(clean["a"])).round(5).tolist(),
        })
        res["coverage"].append(float(seen.sum() / seen.size))

        test_seed = TEST_SEED_BASE + seed * 101

        # --- policies
        pi_bc = AL.fit_bc(clean)
        pi_naive, _, _ = AL.fit_value_iteration(clean, 0.0)
        pi_covi, Q_covi, (n2, chat2, P2, seen2) = AL.fit_value_iteration(clean, ALPHA_STAR)
        Qc = AL.fit_cost_critic(n2, chat2, P2, seen2, BETA_STAR)
        pi_shield = AL.shielded_policy(Q_covi, Qc, TAU_STAR)

        evs = {
            "behaviour": AL.evaluate(None, N_TEST_EP, test_seed,
                                     behaviour=True, eps=BEH_EPS),
            "bc": AL.evaluate(pi_bc, N_TEST_EP, test_seed),
            "naive": AL.evaluate(pi_naive, N_TEST_EP, test_seed),
            "covi": AL.evaluate(pi_covi, N_TEST_EP, test_seed),
            "covi_shield": AL.evaluate(pi_shield, N_TEST_EP, test_seed),
        }
        for m, ev in evs.items():
            s = summarise(ev)
            for k, v in s.items():
                per_seed[m][k].append(v)
            for k in AL.EVENTS:
                event_tot[m][k] += ev["events"][k]
            gap_rate[m].append(ev["gap_unsafe_rate"].tolist())
            gap_occ[m].append(ev["gap_occupancy"].tolist())

        res.setdefault("shield_changes", []).append(int((pi_shield != pi_covi).sum()))

        # --- alpha ablation
        for a in ALPHA_GRID:
            pi_a = (pi_naive if a == 0.0 else
                    (pi_covi if a == ALPHA_STAR else
                     AL.fit_value_iteration(clean, a)[0]))
            ev = AL.evaluate(pi_a, N_TEST_EP_SWEEP, test_seed + 7)
            s = summarise(ev)
            for k in ablation[a]:
                ablation[a][k].append(s[k])

        print(f"  seed {seed} done ({time.time()-t0:.0f}s)", flush=True)

    # ---------------------------------------------------------- aggregate
    for m in methods:
        res["main"][m] = {}
        for k, vals in per_seed[m].items():
            v = np.asarray(vals, float)
            res["main"][m][k] = {
                "mean": float(v.mean()), "std": float(v.std(ddof=1)),
                "ci": boot_ci(v), "per_seed": v.round(4).tolist(),
            }
    res["events"] = event_tot
    res["gap_unsafe_rate"] = {m: np.asarray(v).mean(axis=0).round(5).tolist()
                              for m, v in gap_rate.items()}
    res["gap_occupancy"] = {m: np.asarray(v).mean(axis=0).round(5).tolist()
                            for m, v in gap_occ.items()}
    res["gap_bucket_labels"] = ["critical <12 m", "near 12-30 m",
                                "mid 30-70 m", "far >70 m"]

    for a in ALPHA_GRID:
        res["ablation"][str(a)] = {
            k: {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1))}
            for k, v in ablation[a].items()}

    # ---------------------------------------------------------- statistics
    tests = {}
    fam1 = [("covi", "behaviour"), ("covi", "bc"), ("covi", "naive")]
    for metric in ("return", "crash_rate"):
        raw_p, entries = [], []
        for x, y in fam1:
            t = paired_test(per_seed[x][metric], per_seed[y][metric])
            entries.append((f"{x}_vs_{y}", t)); raw_p.append(t["w_p"])
        for (name, t), adj in zip(entries, holm(raw_p)):
            t["w_p_holm"] = float(adj)
            tests[f"RQ1_{metric}_{name}"] = t

    fam2 = [("covi_shield", "covi")]
    for metric in ("crash_rate", "unsafe_per100", "return"):
        raw_p, entries = [], []
        for x, y in fam2:
            t = paired_test(per_seed[x][metric], per_seed[y][metric])
            entries.append((f"{x}_vs_{y}", t)); raw_p.append(t["w_p"])
        for (name, t), adj in zip(entries, holm(raw_p)):
            t["w_p_holm"] = float(adj)
            tests[f"RQ2_{metric}_{name}"] = t
    res["tests"] = tests

    # ---------------------------------------------------------- robustness
    rob = {m: {s: {"return": [], "crash_rate": [], "unsafe_per100": []}
               for s in NOISE_GRID} for m in ["behaviour", "bc", "naive", "covi"]}
    for seed in ROB_SEEDS:
        raw = AL.collect_dataset(seed=seed, n_episodes=N_TRAIN_EP, eps=BEH_EPS)
        clean, _, _ = AL.filter_dataset(raw)
        pi_bc = AL.fit_bc(clean)
        pi_naive, _, _ = AL.fit_value_iteration(clean, 0.0)
        pi_covi, _, _ = AL.fit_value_iteration(clean, ALPHA_STAR)
        pol = {"bc": pi_bc, "naive": pi_naive, "covi": pi_covi}
        ts = TEST_SEED_BASE + seed * 101 + 13
        for sg in NOISE_GRID:
            for m in rob:
                ev = (AL.evaluate(None, N_TEST_EP_ROB, ts, obs_noise=sg,
                                  behaviour=True, eps=BEH_EPS) if m == "behaviour"
                      else AL.evaluate(pol[m], N_TEST_EP_ROB, ts, obs_noise=sg))
                s = summarise(ev)
                for k in rob[m][sg]:
                    rob[m][sg][k].append(s[k])
        print(f"  robustness seed {seed} done ({time.time()-t0:.0f}s)", flush=True)

    res["robustness"] = {
        m: {str(sg): {k: {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1))}
                      for k, v in d.items()}
            for sg, d in md.items()}
        for m, md in rob.items()}
    res["robustness_seeds"] = ROB_SEEDS
    res["runtime_sec"] = round(time.time() - t0, 1)

    with open("results.json", "w") as f:
        json.dump(res, f, indent=1)
    print(f"\nwrote results.json in {res['runtime_sec']}s")


if __name__ == "__main__":
    main()

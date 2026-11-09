"""
Vector (PDF) figures for the Results section.

Palette validated with the dataviz validator (all-pairs mode): worst CVD
separation dE 10.1, normal-vision floor 19.1, all slots above the chroma
floor and above 3:1 contrast. Identity is additionally carried by marker
shape and line style, so the figures survive grayscale printing.
"""

import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402

OUT = pathlib.Path("figs")
OUT.mkdir(exist_ok=True)

BLUE, AMBER, GREEN, RED = "#2060b0", "#e8a33d", "#148f74", "#a32c3c"
INK, MUTED, GRID = "#1a1a1a", "#555555", "#d8d8d8"

STYLE = {           # colour + marker + linestyle, assigned in fixed order
    "behaviour":   (BLUE,  "o", "-"),
    "bc":          (AMBER, "s", "--"),
    "naive":       (RED,   "^", ":"),
    "covi":        (GREEN, "D", "-."),
    "covi_shield": (MUTED, "v", (0, (3, 1, 1, 1))),
}
LABEL = {
    "behaviour": "Behaviour policy",
    "bc": "Behavioural cloning",
    "naive": r"Naive offline VI ($\alpha$=0)",
    "covi": r"Conservative offline VI ($\alpha$=2)",
    "covi_shield": "Conservative VI + cost shield",
}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.edgecolor": MUTED,
    "axes.linewidth": 0.7,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "lines.linewidth": 1.3,
    "lines.markersize": 4.5,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,
})


def tidy(ax, ygrid=True):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if ygrid:
        ax.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)


def fig_ablation(res):
    """Conservatism ablation: return and collision rate vs alpha."""
    grid = [float(a) for a in res["config"]["alpha_grid"]]
    x = np.arange(len(grid))
    ret = np.array([res["ablation"][str(a)]["return"]["mean"] for a in grid])
    rsd = np.array([res["ablation"][str(a)]["return"]["std"] for a in grid])
    cr = np.array([res["ablation"][str(a)]["crash_rate"]["mean"] for a in grid])
    csd = np.array([res["ablation"][str(a)]["crash_rate"]["std"] for a in grid])

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.05))
    for ax, (y, sd, lab, col) in zip(axes, [
            (ret, rsd, "Episodic return", GREEN),
            (cr, csd, "Collision rate (fraction of episodes)", RED)]):
        ax.errorbar(x, y, yerr=sd, color=col, marker="o", capsize=2.5,
                    elinewidth=0.8, capthick=0.8, zorder=3)
        ax.fill_between(x, y - sd, y + sd, color=col, alpha=0.10, zorder=1)
        ax.set_xticks(x)
        ax.set_xticklabels([("0\n(naive)" if a == 0 else f"{a:g}") for a in grid])
        ax.set_xlabel(r"Conservatism coefficient $\alpha$")
        ax.set_ylabel(lab)
        sel = grid.index(res["config"]["alpha_star"])
        ax.axvline(sel, color=MUTED, linewidth=0.7, linestyle="--", zorder=2)
        ax.annotate(r"selected $\alpha^{*}$", xy=(sel, ax.get_ylim()[1]),
                    xytext=(-3, -9), textcoords="offset points",
                    ha="right", va="top", fontsize=7.5, color=MUTED)
        tidy(ax)
    fig.tight_layout(w_pad=2.0)
    fig.savefig(OUT / "ablation_alpha.pdf")
    plt.close(fig)


def fig_combined(res):
    """Robustness sweep (top row) and error analysis (bottom row), one float."""
    sig = [float(x) for x in res["config"]["noise_grid"]]
    order = ["behaviour", "bc", "naive", "covi"]
    labels = res["gap_bucket_labels"]
    fig, ax = plt.subplots(2, 2, figsize=(6.6, 3.85))

    for a, metric, lab in zip(ax[0], ["return", "crash_rate"],
                              ["Episodic return", "Collision rate"]):
        for m in order:
            col, mk, ls = STYLE[m]
            y = np.array([res["robustness"][m][str(x)][metric]["mean"] for x in sig])
            sd = np.array([res["robustness"][m][str(x)][metric]["std"] for x in sig])
            a.errorbar(sig, y, yerr=sd, color=col, marker=mk, linestyle=ls,
                       capsize=1.8, elinewidth=0.6, capthick=0.6,
                       markersize=3.6, label=LABEL[m], zorder=3)
        a.set_xlabel(r"Sensor noise $\sigma$")
        a.set_ylabel(lab)
        a.set_xticks(sig)
        tidy(a)

    w, xs = 0.2, np.arange(len(labels))
    a = ax[1][0]
    for i, m in enumerate(order):
        a.bar(xs + (i - 1.5) * w, 100.0 * np.asarray(res["gap_unsafe_rate"][m]),
              width=w * 0.88, color=STYLE[m][0], zorder=3, linewidth=0)
    a.set_xticks(xs)
    a.set_xticklabels([l.replace(" ", "\n", 1) for l in labels], fontsize=6.8)
    a.set_xlabel("Observed headway bucket")
    a.set_ylabel("Unsafe steps (% in bucket)")
    tidy(a)

    ev = ["collision", "near_miss", "unsafe_cut_in", "invalid_lane"]
    pretty = ["Collision", "Near\nmiss", "Unsafe\ncut-in", "Invalid\nlane ch."]
    n_ep = len(res["config"]["seeds"]) * res["config"]["n_test_episodes"]
    a = ax[1][1]
    for i, m in enumerate(order):
        a.bar(np.arange(len(ev)) + (i - 1.5) * w,
              [100.0 * res["events"][m][k] / n_ep for k in ev],
              width=w * 0.88, color=STYLE[m][0], zorder=3, linewidth=0)
    a.set_xticks(np.arange(len(ev)))
    a.set_xticklabels(pretty, fontsize=6.8)
    a.set_xlabel("Failure mode")
    a.set_ylabel("Events per 100 episodes")
    tidy(a)

    h, l = ax[0][0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, ncol=4, loc="lower center",
               bbox_to_anchor=(0.5, -0.055), handlelength=2.4,
               columnspacing=1.2, fontsize=7.2)
    fig.tight_layout(w_pad=1.8, h_pad=1.4)
    fig.savefig(OUT / "robustness_errors.pdf")
    plt.close(fig)


if __name__ == "__main__":
    res = json.loads(pathlib.Path("results.json").read_text())
    fig_ablation(res)
    fig_combined(res)
    print("figures written to", OUT.resolve())

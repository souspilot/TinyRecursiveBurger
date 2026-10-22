"""Test-set error by shock strength: does recursion help most on high-amplitude, shock-dominated cases?

    python -m pde_burgers.analyze_by_amplitude --runs 'runs/*_s*' 'runs_extra/shallow4x_s*'

Each test trajectory is binned (equal-count quantile bins) by two features of its TRUE target u(x,t*):
  amplitude      max(u) - min(u)
  max |du/dx|    steepest gradient on the periodic grid (a proxy for shock strength)
For every model, error is reported per bin and averaged over seeds. Runs are grouped by run-directory name
without the `_s<seed>` suffix, so `runs_extra/shallow4x_s0` is its own model, not an extra shallow seed.
"""
import argparse
import glob
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import numpy as np
import torch

from pde_burgers.common import BurgersData, predict
from pde_burgers.evaluate import COLORS, LABELS, _style, load_run

COLORS = {**COLORS, "shallow4x": "#eda100"}
LABELS = {**LABELS, "shallow4x": "2-layer, 4x epochs"}
ORDER = ["shallow", "shallow4x", "recursive", "deep"]
FEATURES = {"amplitude": "amplitude  max(u) − min(u)", "max_grad": "max |∂u/∂x|  (shock strength)"}


def features(ut, x):
    u = ut.double().cpu().numpy()
    dx = float(np.mean(np.diff(x)))
    grad = np.abs(np.roll(u, -1, axis=1) - u) / dx  # periodic forward difference
    return {"amplitude": u.max(1) - u.min(1), "max_grad": grad.max(1)}


def per_traj(pred, true):
    """Per-trajectory relative L2, squared error sum and mean absolute error (physical units)."""
    d = pred.double() - true.double()
    return {"rel_l2": (d.norm(dim=1) / true.double().norm(dim=1)).cpu().numpy(),
            "sse": d.pow(2).sum(1).cpu().numpy(), "mae": d.abs().mean(1).cpu().numpy()}


def bin_stats(pt, bins, n_bins, nx):
    out = {"rel_l2": [], "rmse": [], "mae": [], "sse_share": []}
    total = pt["sse"].sum()
    for b in range(n_bins):
        m = bins == b
        out["rel_l2"].append(pt["rel_l2"][m].mean())
        out["rmse"].append(np.sqrt(pt["sse"][m].sum() / (m.sum() * nx)))
        out["mae"].append(pt["mae"][m].mean())
        out["sse_share"].append(100 * pt["sse"][m].sum() / total)
    return {k: np.array(v) for k, v in out.items()}


def model_key(run_dir):
    m = re.match(r"^(.*)_s(\d+)$", os.path.basename(os.path.normpath(run_dir)))
    return m.group(1) if m else os.path.basename(run_dir)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", default=["runs/*_s*", "runs_extra/shallow4x_s*"])
    p.add_argument("--data_cache", default="data/burgers_nu0.01_t1.npz")
    p.add_argument("--out", default="results_by_amplitude")
    p.add_argument("--bins", type=int, default=4)
    p.add_argument("--bf16", type=lambda s: s.lower() in ("1", "true", "yes"), default=True)
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = BurgersData(args.data_cache, device)
    u0, ut = data.split["test"]
    nx = ut.shape[1]
    feats = features(ut, data.x)
    edges = {f: np.quantile(v, np.linspace(0, 1, args.bins + 1)) for f, v in feats.items()}
    bins = {f: np.clip(np.searchsorted(edges[f][1:-1], v, side="right"), 0, args.bins - 1)
            for f, v in feats.items()}

    run_dirs = sorted({d for g in args.runs for d in glob.glob(g)
                       if "overfit" not in d and os.path.exists(os.path.join(d, "best_ema.pt"))
                       and os.path.exists(os.path.join(d, "summary.json"))})
    print(f"[bins] test n={len(u0)}; runs: {run_dirs}")

    # stats[feature][model] -> list over seeds of bin_stats; steps[feature] -> list over seeds of [K x bins]
    stats = {f: defaultdict(list) for f in feats}
    steps = {f: [] for f in feats}
    for rd in run_dirs:
        _, model, _ = load_run(rd, device)
        key = model_key(rd)
        preds = predict(model, data, u0, device, args.bf16)
        for f in feats:
            stats[f][key].append(bin_stats(per_traj(preds[-1], ut), bins[f], args.bins, nx))
            if model.recursive:
                steps[f].append(np.stack([bin_stats(per_traj(pk, ut), bins[f], args.bins, nx)["rel_l2"]
                                          for pk in preds]))
        print(f"[bins] {rd} -> {key}")

    models = [m for m in ORDER if m in stats["amplitude"]] + \
             sorted(m for m in stats["amplitude"] if m not in ORDER)
    write_tables(args, feats, edges, stats, steps, models)
    plot(args, edges, stats, models)


def ms(arrs, key):
    a = np.stack([s[key] for s in arrs])
    return a.mean(0), (a.std(0, ddof=1) if len(a) > 1 else None)


def cell(m, s, f):
    return f.format(m) + (f" ± {f.format(s)}" if s is not None else "")


def write_tables(args, feats, edges, stats, steps, models):
    lines = ["# Test error by shock strength", "",
             f"Test trajectories split into {args.bins} equal-count bins of each feature of the true target "
             "u(x,t*). Mean ± sample std over seeds. Bin 1 = weakest, last bin = strongest.", ""]
    for f, label in FEATURES.items():
        e = edges[f]
        hdr = [f"bin {b + 1}: {e[b]:.3g}–{e[b + 1]:.3g}" for b in range(args.bins)]
        lines += [f"## By {label}", ""]
        for metric, title, fmt in (("rel_l2", "Mean relative L²", "{:.4f}"), ("rmse", "RMSE", "{:.4f}"),
                                   ("sse_share", "Share of the model's total squared error (%)", "{:.1f}")):
            lines += [f"**{title}**", "", "| model | seeds | " + " | ".join(hdr) + " |",
                      "|" + "---|" * (2 + args.bins)]
            for mdl in models:
                m, s = ms(stats[f][mdl], metric)
                lines.append(f"| {LABELS.get(mdl, mdl)} | {len(stats[f][mdl])} | " +
                             " | ".join(cell(m[b], None if s is None else s[b], fmt) for b in range(args.bins)) + " |")
            lines.append("")
        if steps[f]:
            a = np.stack(steps[f])  # seeds x K x bins
            lines += ["**Recursive model, mean relative L² after each refinement step**", "",
                      "| step | " + " | ".join(hdr) + " |", "|" + "---|" * (1 + args.bins)]
            for k in range(a.shape[1]):
                sd = a[:, k].std(0, ddof=1) if len(a) > 1 else [None] * args.bins
                lines.append(f"| {k + 1} | " + " | ".join(cell(a[:, k, b].mean(), sd[b], "{:.4f}")
                                                         for b in range(args.bins)) + " |")
            lines.append("")
    with open(os.path.join(args.out, "by_amplitude.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def plot(args, edges, stats, models):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    pos = np.arange(args.bins)
    for i, (f, label) in enumerate(FEATURES.items()):
        e = edges[f]
        ticks = [f"Q{b + 1}\n{e[b]:.3g}–{e[b + 1]:.3g}" for b in range(args.bins)]
        for j, (metric, ylab) in enumerate((("rel_l2", "mean relative L2"), ("rmse", "RMSE"))):
            ax = axes[i, j]
            for mdl in models:
                m, s = ms(stats[f][mdl], metric)
                ax.errorbar(pos, m, yerr=s, color=COLORS.get(mdl, "#52514e"), marker="o", ms=7, lw=2,
                            capsize=3, label=LABELS.get(mdl, mdl))
            ax.set_xticks(pos, ticks, fontsize=8)
            ax.set_xlabel(f"{label}: equal-count bins, weakest → strongest")
            ax.set_ylabel(ylab)
            _style(ax)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(os.path.join(args.out, "error_by_shock_strength.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()

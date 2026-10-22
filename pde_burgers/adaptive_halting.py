"""Inference-time adaptive halting for the trained recursive model (no retraining).

    python -m pde_burgers.adaptive_halting --runs 'runs/recursive_s*' \
        --refs results_prov/metrics.json results_shallow4x/metrics.json

Each input gets a difficulty score that is available at inference time:
  pred_grad  max |d/dx| of the step-1 prediction (free: step 1 must run anyway)
  u0_grad    max |d/dx| of the input u(x,0)
The halting policy is fitted on VALIDATION only: split validation trajectories into equal-count bins of the
score and pick, per bin, the refinement step with the lowest mean relative L2. The bin edges and step choices
are then applied once to the test set. Also reported: every fixed step, and a per-trajectory oracle (best step
chosen with the test target, an upper bound that is not achievable).
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import numpy as np
import torch

from pde_burgers.common import BurgersData, predict
from pde_burgers.evaluate import COLORS, _style, load_run
from pde_burgers.models import flops_per_sample

SCORES = {"pred_grad": "max |∂û/∂x| of the step-1 prediction", "u0_grad": "max |∂u/∂x| of the input u(x,0)"}


def max_grad(u, dx):
    u = u.double()
    return ((torch.roll(u, -1, dims=1) - u).abs().max(1).values / dx).cpu().numpy()


def rel_l2(pred, true):
    return ((pred.double() - true.double()).norm(dim=1) / true.double().norm(dim=1)).cpu().numpy()


def sse(pred, true):
    return (pred.double() - true.double()).pow(2).sum(1).cpu().numpy()


def split_preds(model, data, split, device, bf16):
    u0, ut = data.split[split]
    preds = predict(model, data, u0, device, bf16)
    dx = float(np.mean(np.diff(data.x)))
    return {"rel": np.stack([rel_l2(p, ut) for p in preds]),  # K x n
            "sse": np.stack([sse(p, ut) for p in preds]),
            "score": {"pred_grad": max_grad(preds[0], dx), "u0_grad": max_grad(u0, dx)},
            "nx": ut.shape[1]}


def fit_policy(val, score, n_bins):
    s = val["score"][score]
    edges = np.quantile(s, np.linspace(0, 1, n_bins + 1))[1:-1]
    b = np.searchsorted(edges, s, side="right")
    steps = [int(val["rel"][:, b == i].mean(1).argmin()) for i in range(n_bins)]
    return edges, steps


def apply_policy(test, score, edges, steps):
    b = np.searchsorted(edges, test["score"][score], side="right")
    k = np.array(steps)[b]  # chosen step index per trajectory
    idx = np.arange(test["rel"].shape[1])
    return k, test["rel"][k, idx], test["sse"][k, idx]


def summarize(rel, sse_, k, nx, flops_step):
    return {"rel_l2": float(rel.mean()), "rmse": float(np.sqrt(sse_.sum() / (len(rel) * nx))),
            "mean_steps": float(k.mean() + 1), "mean_mflops": float(np.mean(np.array(flops_step)[k]) / 1e6)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="runs/recursive_s*")
    p.add_argument("--data_cache", default="data/burgers_nu0.01_t1.npz")
    p.add_argument("--out", default="results_adaptive")
    p.add_argument("--bins", type=int, default=4)
    p.add_argument("--refs", nargs="*", default=[], help="metrics.json files whose shallow/deep rows to draw")
    p.add_argument("--bf16", type=lambda s: s.lower() in ("1", "true", "yes"), default=True)
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = BurgersData(args.data_cache, device)

    run_dirs = sorted(d for d in glob.glob(args.runs) if os.path.exists(os.path.join(d, "best_ema.pt"))
                      and os.path.exists(os.path.join(d, "summary.json")) and "overfit" not in d)
    print(f"[halt] runs: {run_dirs}")
    per_seed = []
    for rd in run_dirs:
        ck, model, _ = load_run(rd, device)
        assert model.recursive, f"{rd} is not a recursive model"
        cfg = ck["cfg"]
        flops_step = [flops_per_sample("recursive", cfg, n_steps=k) for k in range(1, model.n_sup + 1)]
        val = split_preds(model, data, "val", device, args.bf16)
        test = split_preds(model, data, "test", device, args.bf16)
        n, K = test["rel"].shape[1], test["rel"].shape[0]
        res = {"run": rd, "fixed": [], "policies": {}}
        for k in range(K):
            kk = np.full(n, k)
            res["fixed"].append(summarize(test["rel"][k], test["sse"][k], kk, test["nx"], flops_step))
        for sc in SCORES:
            edges, steps = fit_policy(val, sc, args.bins)
            k, rel, s = apply_policy(test, sc, edges, steps)
            res["policies"][sc] = {**summarize(rel, s, k, test["nx"], flops_step),
                                   "val_edges": edges.tolist(), "steps_per_bin": [x + 1 for x in steps]}
        ko = test["rel"].argmin(0)
        idx = np.arange(n)
        res["oracle"] = summarize(test["rel"][ko, idx], test["sse"][ko, idx], ko, test["nx"], flops_step)
        per_seed.append(res)
        print(f"[halt] {rd}: fixed step {K} relL2 {res['fixed'][-1]['rel_l2']:.5f}; "
              + "; ".join(f"{sc} {v['rel_l2']:.5f} (steps/bin {v['steps_per_bin']})"
                          for sc, v in res["policies"].items()))
    if not per_seed:
        raise SystemExit("no finished recursive runs found")

    refs = load_refs(args.refs)
    with open(os.path.join(args.out, "adaptive.json"), "w") as f:
        json.dump({"seeds": per_seed, "refs": refs}, f, indent=1)
    write_table(args, per_seed, refs)
    plot(args, per_seed, refs)


def load_refs(paths):
    """Final-output test relative L2 of shallow / deep rows from evaluate.py metrics.json files."""
    refs = {}
    for pth in paths:
        rows = json.load(open(pth))["rows"]
        for name, label in (("shallow", "2-layer"), ("deep", "24-layer untied")):
            vals = [r["test"]["rel_l2_mean"] for r in rows if r["model"] == name]
            if not vals:
                continue
            if "shallow4x" in pth:  # results_shallow4x/metrics.json labels its runs "shallow"
                name, label = "shallow4x", "2-layer, 4x epochs"
            refs[name] = {"label": label, "rel_l2": float(np.mean(vals))}
    return refs


def agg(per_seed, get):
    v = np.array([get(r) for r in per_seed])
    return v.mean(), (v.std(ddof=1) if len(v) > 1 else None)


def fmt(m, s, f="{:.4f}"):
    return f.format(m) + (f" ± {f.format(s)}" if s is not None else "")


def write_table(args, per_seed, refs):
    K = len(per_seed[0]["fixed"])
    lines = ["# Adaptive halting (inference-time, no retraining)", "",
             f"Recursive model, {len(per_seed)} seed(s), mean ± std. Policies were fitted on validation "
             f"({args.bins} equal-count bins of the difficulty score; best step per bin) and applied once to test.", "",
             "| output | test rel L2 | test RMSE | mean steps | mean MFLOPs/sample |", "|---|---:|---:|---:|---:|"]
    row = lambda name, get: lines.append(
        f"| {name} | {fmt(*agg(per_seed, lambda r: get(r)['rel_l2']))} | {fmt(*agg(per_seed, lambda r: get(r)['rmse']))} | "
        f"{fmt(*agg(per_seed, lambda r: get(r)['mean_steps']), '{:.2f}')} | "
        f"{fmt(*agg(per_seed, lambda r: get(r)['mean_mflops']), '{:.0f}')} |")
    for k in range(K):
        row(f"fixed step {k + 1}" + (" (reported in main results)" if k == K - 1 else ""), lambda r, k=k: r["fixed"][k])
    for sc, label in SCORES.items():
        row(f"**adaptive**, score = {label}", lambda r, sc=sc: r["policies"][sc])
    row("oracle: best step per trajectory, chosen with the test target (upper bound)", lambda r: r["oracle"])
    lines += ["", "Steps chosen per validation bin (weakest → strongest score), per seed:", ""]
    for r in per_seed:
        lines.append(f"* `{r['run']}`: " + "; ".join(f"{sc}: {v['steps_per_bin']}" for sc, v in r["policies"].items()))
    if refs:
        lines += ["", "Reference (final output, test rel L2): " +
                  ", ".join(f"{v['label']} {v['rel_l2']:.4f}" for v in refs.values())]
    with open(os.path.join(args.out, "adaptive.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def plot(args, per_seed, refs):
    K = len(per_seed[0]["fixed"])
    fig, ax = plt.subplots(figsize=(8, 5))
    xs = [agg(per_seed, lambda r, k=k: r["fixed"][k]["mean_mflops"])[0] for k in range(K)]
    ys = [agg(per_seed, lambda r, k=k: r["fixed"][k]["rel_l2"]) for k in range(K)]
    ax.errorbar(xs, [m for m, _ in ys], yerr=[s for _, s in ys] if ys[0][1] is not None else None,
                color=COLORS["recursive"], marker="o", ms=8, lw=2, capsize=3, label="recursive, fixed steps")
    for k, (x, (y, _)) in enumerate(zip(xs, ys)):
        ax.annotate(f"step {k + 1}", (x, y), textcoords="offset points", xytext=(6, 6), fontsize=8, color="#52514e")
    marks = {"pred_grad": ("D", "adaptive: step-1 prediction's max |∂û/∂x|"), "u0_grad": ("s", "adaptive: input's max |∂u/∂x|")}
    for sc, (mk, label) in marks.items():
        x = agg(per_seed, lambda r: r["policies"][sc]["mean_mflops"])[0]
        y, s = agg(per_seed, lambda r: r["policies"][sc]["rel_l2"])
        ax.errorbar([x], [y], yerr=[s] if s is not None else None, color="#a8401a", marker=mk, ms=9, lw=0,
                    elinewidth=1.5, capsize=3, label=label)
    x = agg(per_seed, lambda r: r["oracle"]["mean_mflops"])[0]
    ax.scatter([x], [agg(per_seed, lambda r: r["oracle"]["rel_l2"])[0]], marker="*", s=160, color="#8a8984",
               label="oracle per trajectory (upper bound)", zorder=3)
    ref_colors = {"shallow": COLORS["shallow"], "shallow4x": "#eda100", "deep": COLORS["deep"]}
    for name, v in refs.items():
        ax.axhline(v["rel_l2"], color=ref_colors.get(name, "#52514e"), lw=1.5, label=v["label"])
    ax.set_xlabel("mean MFLOPs per sample")
    ax.set_ylabel("test mean relative L2")
    _style(ax)
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "adaptive_halting.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()

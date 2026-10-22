"""Test-set evaluation of selected (best-validation EMA) checkpoints, latency, and plots.

    python -m pde_burgers.evaluate --runs 'runs/*_s*' --out results

Run on an otherwise idle GPU so latencies are comparable. Test data is touched only here.
"""
import argparse
import glob
import json
import os
import statistics
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from pde_burgers.common import BurgersData, autocast, metrics, per_traj_rel_l2, predict
from pde_burgers.models import build_model, core_layer_calls, count_params

COLORS = {"shallow": "#2a78d6", "recursive": "#eb6834", "deep": "#1baf7a", "persistence": "#8a8984",
          "ridge": "#52514e"}
LABELS = {"shallow": "2-layer (no recursion)", "recursive": "recursive TRM (2-layer core)",
          "deep": "24-layer untied", "persistence": "persistence u(x,0)",
          "mean": "training-mean field", "ridge": "ridge regression (linear)"}
ORDER = ["mean", "persistence", "ridge", "shallow", "recursive", "deep"]
RIDGE_ALPHAS = [1e-2, 1e-1, 1, 10, 100, 1e3, 1e4]


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def latency_ms(model, data, u0, device, bf16, n_steps=None, warmup=10, reps=50):
    """Median wall-clock ms for one forward pass on the batch `u0` (normalisation included)."""
    model.eval()
    def run():
        with autocast(device, bf16):
            x = data.norm(u0)
            return model(x, n_steps=n_steps) if model.recursive else model(x)
    for _ in range(warmup):
        run()
    ts = []
    for _ in range(reps):
        sync(device)
        t0 = time.perf_counter()
        run()
        sync(device)
        ts.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(ts)


def load_run(run_dir, device):
    ck = torch.load(os.path.join(run_dir, "best_ema.pt"), map_location=device)
    model = build_model(ck["model_name"], ck["cfg"]).to(device)
    model.load_state_dict(ck["state_dict"])
    summ_path = os.path.join(run_dir, "summary.json")
    summary = json.load(open(summ_path)) if os.path.exists(summ_path) else {}
    return ck, model, summary


def reference_baselines(data):
    """Untrained/closed-form reference points, fit on train only (ridge alpha chosen on val).

    mean : the training-set average u(x,t*), ignoring the input -> tests whether u(x,0) is used at all.
    ridge: best linear map u(x,0) -> u(x,t*) (with bias) -> tests whether nonlinearity is needed.
    """
    dev = data.split["train"][0].device
    (u0_tr, ut_tr), (u0_va, ut_va), (u0_te, _) = (data.split[s] for s in ("train", "val", "test"))
    mean_field = ut_tr.double().mean(0)
    out = {"mean": {"fit": {"val": metrics(mean_field.expand_as(ut_va), ut_va)},
                    "pred": lambda u: mean_field.expand(len(u), -1).float(), "params": mean_field.numel()}}

    X = torch.cat([data.norm(u0_tr).double(), torch.ones(len(u0_tr), 1, dtype=torch.float64, device=dev)], 1)
    Y = data.norm(ut_tr).double()
    XtX, XtY = X.T @ X, X.T @ Y
    eye = torch.eye(X.shape[1], dtype=torch.float64, device=dev)
    lin = lambda W, u: data.denorm(torch.cat([data.norm(u).double(), torch.ones(len(u), 1, dtype=torch.float64,
                                                                                  device=dev)], 1) @ W).float()
    best = None
    for a in RIDGE_ALPHAS:
        W = torch.linalg.solve(XtX + a * eye, XtY)
        val = metrics(lin(W, u0_va), ut_va)
        print(f"[eval] ridge alpha={a:g}: val relL2 {val['rel_l2_mean']:.5f}")
        if best is None or val["rel_l2_mean"] < best[1]["rel_l2_mean"]:
            best = (a, val, W)
    a, val, W = best
    out["ridge"] = {"fit": {"val": val, "alpha": a}, "pred": lambda u: lin(W, u), "params": W.numel()}
    return out


def mean_std(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None
    return float(np.mean(vals)), (float(np.std(vals, ddof=1)) if len(vals) > 1 else None)


def fmt(m, s, f="{:.5f}"):
    if m is None:
        return "-"
    return f.format(m) + (f" ± {f.format(s)}" if s is not None else "")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="runs/*_s*")
    p.add_argument("--data_cache", default="data/burgers_nu0.01_t1.npz")
    p.add_argument("--out", default="results")
    p.add_argument("--bf16", type=lambda s: s.lower() in ("1", "true", "yes"), default=True)
    p.add_argument("--latency_batches", type=int, nargs="+", default=[1, 1000])
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = BurgersData(args.data_cache, device)
    u0, ut = data.split["test"]
    run_dirs = sorted(d for d in glob.glob(args.runs)
                      if os.path.exists(os.path.join(d, "best_ema.pt")) and "overfit" not in d)
    print(f"[eval] test n={len(u0)}; runs: {run_dirs}")

    rows = [{"model": "persistence", "run": "-", "seed": None, "params": 0, "test": metrics(u0, ut),
             "latency_ms": {}, "core_layer_calls": 0}]
    pertraj = {"persistence": per_traj_rel_l2(u0, ut)}
    preds_for_plot = {"persistence": u0}
    for name, ref in reference_baselines(data).items():
        pred = ref["pred"](u0)
        rows.append({"model": name, "run": "-", "seed": None, "params": ref["params"], **ref["fit"],
                     "test": metrics(pred, ut), "latency_ms": {}, "core_layer_calls": 0})
        pertraj[name], preds_for_plot[name] = per_traj_rel_l2(pred, ut), pred
        print(f"[eval] {name}: test relL2 {rows[-1]['test']['rel_l2_mean']:.5f}")

    for rd in run_dirs:
        ck, model, summary = load_run(rd, device)
        name, cfg = ck["model_name"], ck["cfg"]
        preds = predict(model, data, u0, device, args.bf16)
        row = {"model": name, "run": rd, "seed": cfg["seed"], "params": count_params(model),
               "selected_epoch": ck["epoch"], "selected_step": ck["step"], "val": ck["val"],
               "test": metrics(preds[-1], ut), "train_sec": summary.get("train_sec"),
               "optimizer_steps": summary.get("optimizer_steps"), "epochs_done": summary.get("epochs_done"),
               "core_layer_calls": core_layer_calls(name, cfg), "latency_ms": {}}
        for bs in args.latency_batches:
            xb = u0[:bs]
            if model.recursive:
                row["latency_ms"][bs] = {k: latency_ms(model, data, xb, device, args.bf16, n_steps=k)
                                         for k in range(1, model.n_sup + 1)}
            else:
                row["latency_ms"][bs] = latency_ms(model, data, xb, device, args.bf16)
        if model.recursive:
            row["test_per_step"] = [metrics(p_, ut) for p_ in preds]
        rows.append(row)
        if name not in pertraj:  # plots use the lowest-seed run of each model (run_dirs are sorted)
            pertraj[name], preds_for_plot[name] = per_traj_rel_l2(preds[-1], ut), preds[-1]
            if model.recursive:
                preds_for_plot["recursive_steps"] = preds
        print(f"[eval] {rd}: test relL2 {row['test']['rel_l2_mean']:.5f} rmse {row['test']['rmse']:.5f} "
              f"latency {row['latency_ms']}")

    out = {"data": {k: data.meta[k] for k in ("target_time_actual", "target_index", "input_time_actual",
                                              "time_convention", "split_sizes", "norm_mean", "norm_std")},
           "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
           "bf16": args.bf16, "rows": rows}
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(out, f, indent=1, default=str)

    write_table(rows, args, out)
    plot_fields(data, u0, ut, preds_for_plot, pertraj, args.out)
    plot_refinement(rows, args)
    print(f"[eval] wrote {args.out}/metrics.json, results.md and figures")


def final_latency(row, bs):
    lat = row["latency_ms"].get(bs)
    return max(lat.items())[1] if isinstance(lat, dict) else lat


def write_table(rows, args, out):
    by = defaultdict(list)
    for r in rows:
        by[r["model"]].append(r)
    lb = args.latency_batches
    lines = [f"Test set: n={out['data']['split_sizes']['test']} trajectories, target t={out['data']['target_time_actual']}"
             f" (snapshot {out['data']['target_index']}), device {out['device']}, bf16={out['bf16']}.",
             "Mean ± sample std over seeds when more than one seed exists.", "",
             "| model | seeds | params | core layer calls | val relL2 | test relL2 (mean) | test RMSE | "
             + " | ".join(f"latency bs={b} (ms)" for b in lb) + " | train time (min) | optimizer steps |",
             "|" + "---|" * (9 + len(lb))]
    for name in ORDER:
        if name not in by:
            continue
        rs = by[name]
        g = lambda f: mean_std([f(r) for r in rs])
        lat = [fmt(*g(lambda r, b=b: final_latency(r, b)), "{:.2f}") for b in lb]
        lines.append(f"| {LABELS[name]} | {len([r for r in rs if r['seed'] is not None]) or '-'} | {rs[0]['params']:,} | "
                     f"{rs[0]['core_layer_calls']} | {fmt(*g(lambda r: r.get('val', {}).get('rel_l2_mean')))} | "
                     f"{fmt(*g(lambda r: r['test']['rel_l2_mean']))} | {fmt(*g(lambda r: r['test']['rmse']))} | "
                     + " | ".join(lat) +
                     f" | {fmt(*g(lambda r: r['train_sec'] / 60 if r.get('train_sec') else None), '{:.1f}')} | "
                     f"{fmt(*g(lambda r: r.get('optimizer_steps')), '{:.0f}')} |")
    if "recursive" in by:
        lines += ["", "Recursive model after each refinement step (test, same checkpoint):", "",
                  "| step | core layer calls | test relL2 | test RMSE | " + " | ".join(f"latency bs={b} (ms)" for b in lb) + " |",
                  "|" + "---|" * (4 + len(lb))]
        rs = by["recursive"]
        cfgl = rs[0]["core_layer_calls"] // len(rs[0]["test_per_step"])
        for k in range(len(rs[0]["test_per_step"])):
            g = lambda f: mean_std([f(r) for r in rs])
            lines.append(f"| {k + 1} | {cfgl * (k + 1)} | {fmt(*g(lambda r: r['test_per_step'][k]['rel_l2_mean']))} | "
                         f"{fmt(*g(lambda r: r['test_per_step'][k]['rmse']))} | "
                         + " | ".join(fmt(*g(lambda r, b=b: r['latency_ms'][b][k + 1]), '{:.2f}') for b in lb) + " |")
    with open(os.path.join(args.out, "results.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25, linewidth=0.6)


def plot_fields(data, u0, ut, preds, pertraj, out_dir):
    """Three test trajectories at the 10th/50th/90th percentile of the reference model's error."""
    ref = "recursive" if "recursive" in pertraj else ("shallow" if "shallow" in pertraj else "persistence")
    err = pertraj[ref]
    picks = [int(np.argsort(err)[int(q * (len(err) - 1))]) for q in (0.1, 0.5, 0.9)]
    x = data.x
    models = [m for m in ("ridge", "shallow", "recursive", "deep") if m in preds]
    fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharex=True)
    for j, (i, q) in enumerate(zip(picks, ("p10", "median", "p90"))):
        ax = axes[0, j]
        ax.plot(x, u0[i].cpu(), color=COLORS["persistence"], lw=1.2, ls="--", label="u(x,0)")
        ax.plot(x, ut[i].cpu(), color="#0b0b0b", lw=2, label="true u(x,t*)")
        for m in models:
            ax.plot(x, preds[m][i].cpu(), color=COLORS[m], lw=1.4, label=LABELS[m],
                    ls=":" if m == "ridge" else "-")
        ax.set_title(f"test #{data.idx['test'][i].item()} ({q} of {ref} error)")
        _style(ax)
        ax2 = axes[1, j]
        for m in models:
            ax2.plot(x, (preds[m][i] - ut[i]).abs().cpu(), color=COLORS[m], lw=1.2, label=LABELS[m],
                     ls=":" if m == "ridge" else "-")
        ax2.set_yscale("log")
        ax2.set_xlabel("x")
        _style(ax2)
    axes[0, 0].set_ylabel("u")
    axes[1, 0].set_ylabel("|prediction − truth|")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(os.path.join(out_dir, "fields.png"), dpi=150)
    plt.close(fig)

    if "recursive_steps" in preds:
        fig, axes = plt.subplots(1, 3, figsize=(15, 3.5), sharey=True)
        steps = preds["recursive_steps"]
        shades = ["#f7c2a8", "#f29a70", "#eb6834", "#a8401a"][-len(steps):]
        for j, i in enumerate(picks):
            for k, p_ in enumerate(steps):
                axes[j].plot(data.x, (p_[i] - ut[i]).abs().cpu(), color=shades[k], lw=1.2, label=f"step {k + 1}")
            axes[j].set_yscale("log")
            axes[j].set_xlabel("x")
            _style(axes[j])
        axes[0].set_ylabel("|error| (recursive)")
        axes[0].legend(fontsize=8, frameon=False)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "recursive_spatial_error_by_step.png"), dpi=150)
        plt.close(fig)


def plot_refinement(rows, args):
    by = defaultdict(list)
    for r in rows:
        by[r["model"]].append(r)
    bs = max(args.latency_batches)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.5))
    if "recursive" in by:
        rs = by["recursive"]
        K = len(rs[0]["test_per_step"])
        ks = np.arange(1, K + 1)
        e = np.array([[r["test_per_step"][k]["rel_l2_mean"] for k in range(K)] for r in rs])
        lat = np.array([[r["latency_ms"][bs][k + 1] for k in range(K)] for r in rs])
        for arr_x, ax in ((ks, a1), (lat.mean(0), a2)):
            ax.errorbar(arr_x, e.mean(0), yerr=e.std(0, ddof=1) if len(rs) > 1 else None,
                        color=COLORS["recursive"], marker="o", ms=8, lw=2, capsize=3, label=LABELS["recursive"])
        for k in ks:
            a2.annotate(f"step {k}", (lat.mean(0)[k - 1], e.mean(0)[k - 1]), textcoords="offset points",
                        xytext=(6, 6), fontsize=8, color="#52514e")
    # persistence/ridge are orders of magnitude worse and would flatten these axes; they live in results.md
    for name in ("shallow", "deep"):
        if name not in by:
            continue
        m = np.mean([r["test"]["rel_l2_mean"] for r in by[name]])
        a1.axhline(m, color=COLORS[name], lw=1.5, label=LABELS[name])
        a2.scatter([np.mean([final_latency(r, bs) for r in by[name]])], [m], s=70,
                   color=COLORS[name], edgecolor="white", linewidth=2, zorder=3, label=LABELS[name])
    a1.set_xlabel("refinement step")
    a1.set_ylabel("test mean relative L2")
    a1.set_xticks(range(1, 5))
    a2.set_xlabel(f"inference latency, batch {bs} (ms)")
    a2.set_ylabel("test mean relative L2")
    for ax in (a1, a2):
        _style(ax)
        ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "error_vs_refinement_and_latency.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()

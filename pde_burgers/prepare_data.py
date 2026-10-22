"""Inspect the PDEBench Burgers HDF5, extract (u(x,0), u(x,t*)) pairs, split, and normalise.

Order of operations (deliberate):
  1. inspect + log keys, shapes, attrs, coordinates
  2. pick t* = stored time nearest --target_time
  3. split trajectory indices 80/10/10 with seed 42 (before any preprocessing)
  4. fit ONE scalar mean/std on the training split (inputs and targets pooled)
The cache stores raw physical-unit fields; normalisation is applied in-model-pipeline
using the stored training statistics so evaluation can invert it exactly.
"""
import argparse
import json
import os

import h5py
import numpy as np

SPLIT_SEED = 42


def describe(f):
    info = {"root_attrs": {k: _jsonable(v) for k, v in f.attrs.items()}, "datasets": {}}

    def visit(name, obj):
        if isinstance(obj, h5py.Dataset):
            info["datasets"][name] = {"shape": list(obj.shape), "dtype": str(obj.dtype),
                                      "chunks": obj.chunks,
                                      "attrs": {k: _jsonable(v) for k, v in obj.attrs.items()}}
    f.visititems(visit)
    return info


def _jsonable(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.generic,)):
        return v.item()
    if isinstance(v, bytes):
        return v.decode(errors="replace")
    return v


def snapshot_times(t_coord, nt, convention):
    """PDEBench 1D files store t-coordinate with nt+1 entries for nt snapshots.
    'leading' : snapshot k is at t_coord[k] (snapshot 0 = initial condition at t_coord[0]).
    'trailing': snapshot k is at t_coord[k+1].
    """
    if len(t_coord) == nt:
        return t_coord, "exact (len(t)==nt)"
    if len(t_coord) == nt + 1:
        return (t_coord[:nt], "leading") if convention == "leading" else (t_coord[1:], "trailing")
    raise ValueError(f"cannot reconcile t-coordinate length {len(t_coord)} with nt={nt}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--h5", default="data/pdebench/1D/Burgers/Train/1D_Burgers_Sols_Nu0.01.hdf5")
    p.add_argument("--out", default="data/burgers_nu0.01_t1.npz")
    p.add_argument("--target_time", type=float, default=1.0)
    p.add_argument("--time_convention", choices=["leading", "trailing"], default="leading")
    p.add_argument("--read_block", type=int, default=500)
    args = p.parse_args()

    with h5py.File(args.h5, "r") as f:
        info = describe(f)
        print(json.dumps(info, indent=1, default=str))
        if "tensor" not in f:
            raise SystemExit(f"expected dataset 'tensor'; found {list(info['datasets'])}")
        ds = f["tensor"]
        n_traj, nt, nx = ds.shape
        x = np.asarray(f["x-coordinate"][:], dtype=np.float64)
        t_coord = np.asarray(f["t-coordinate"][:], dtype=np.float64)

        times, conv = snapshot_times(t_coord, nt, args.time_convention)
        k = int(np.argmin(np.abs(times - args.target_time)))
        log = {
            "h5": os.path.abspath(args.h5), "tensor_shape": [n_traj, nt, nx],
            "x_len": len(x), "x_min": x.min(), "x_max": x.max(), "dx": float(np.diff(x).mean()),
            "t_coord_len": len(t_coord), "t_coord_first5": t_coord[:5].tolist(),
            "t_coord_last5": t_coord[-5:].tolist(),
            "time_convention": conv, "target_time_requested": args.target_time,
            "target_index": k, "target_time_actual": float(times[k]),
            "input_index": 0, "input_time_actual": float(times[0]),
            "h5_info": info,
        }
        print(f"[data] tensor {ds.shape}; t-coordinate has {len(t_coord)} entries "
              f"({t_coord[0]}..{t_coord[-1]}); convention={conv}")
        print(f"[data] input = snapshot 0 (t={times[0]:.4f}); target = snapshot {k} (t={times[k]:.4f})")
        if conv in ("leading", "trailing"):
            alt = "trailing" if conv == "leading" else "leading"
            alt_times, _ = snapshot_times(t_coord, nt, alt)
            log["alt_convention_target_time"] = float(alt_times[k])
            print(f"[data] NOTE: under the '{alt}' convention snapshot {k} would be t={alt_times[k]:.4f}")

        assert nx == 1024, f"expected 1024 spatial points, got {nx}"
        u0 = np.empty((n_traj, nx), np.float32)
        ut = np.empty((n_traj, nx), np.float32)
        for s in range(0, n_traj, args.read_block):
            e = min(s + args.read_block, n_traj)
            blk = ds[s:e, [0, k], :]
            u0[s:e], ut[s:e] = blk[:, 0], blk[:, 1]
            print(f"\r[data] read {e}/{n_traj}", end="", flush=True)
        print()

    # split BEFORE preprocessing
    perm = np.random.default_rng(SPLIT_SEED).permutation(n_traj)
    n_tr, n_va = int(0.8 * n_traj), int(0.1 * n_traj)
    idx = {"train": np.sort(perm[:n_tr]), "val": np.sort(perm[n_tr:n_tr + n_va]),
           "test": np.sort(perm[n_tr + n_va:])}

    # scalar normalisation from training data only
    tr = np.concatenate([u0[idx["train"]].ravel(), ut[idx["train"]].ravel()]).astype(np.float64)
    mean, std = float(tr.mean()), float(tr.std())
    log.update({"split_seed": SPLIT_SEED, "split_sizes": {k_: len(v) for k_, v in idx.items()},
                "norm_mean": mean, "norm_std": std,
                "stats_u0": [float(u0.min()), float(u0.max())], "stats_ut": [float(ut.min()), float(ut.max())],
                "nonfinite": int((~np.isfinite(u0)).sum() + (~np.isfinite(ut)).sum())})
    print(f"[data] split {log['split_sizes']}; norm mean={mean:.6f} std={std:.6f}; nonfinite={log['nonfinite']}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez(args.out, u0=u0, ut=ut, x=x, train_idx=idx["train"], val_idx=idx["val"],
             test_idx=idx["test"], norm_mean=mean, norm_std=std, meta=json.dumps(log, default=str))
    with open(os.path.splitext(args.out)[0] + "_meta.json", "w") as fh:
        json.dump(log, fh, indent=1, default=str)
    print(f"[data] wrote {args.out}")


if __name__ == "__main__":
    main()

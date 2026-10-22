"""Config, data, prediction and metric helpers shared by training and evaluation.

Preprocessing and metrics live here only, so every model is treated identically.
"""
import argparse
import contextlib
import json
import os

import numpy as np
import torch
import yaml

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")


def load_config(argv=None, extra_args=()):
    """Defaults from config.yaml; every key becomes an overridable --flag."""
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    p = argparse.ArgumentParser()
    for name, kwargs in extra_args:
        p.add_argument(name, **kwargs)
    for k, v in cfg.items():
        if isinstance(v, bool):
            p.add_argument(f"--{k}", type=lambda s: s.lower() in ("1", "true", "yes"), default=v)
        else:
            p.add_argument(f"--{k}", type=type(v), default=v)
    args = vars(p.parse_args(argv))
    return {k: args[k] for k in cfg}, {k: v for k, v in args.items() if k not in cfg}


class BurgersData:
    """Fields kept in physical units on `device`; models see (u - mean) / std."""

    def __init__(self, path, device, overfit_n=0):
        d = np.load(path)
        self.meta = json.loads(str(d["meta"]))
        self.mean, self.std = float(d["norm_mean"]), float(d["norm_std"])
        self.x = d["x"]
        u0, ut = torch.from_numpy(d["u0"]), torch.from_numpy(d["ut"])
        self.idx = {s: torch.from_numpy(d[f"{s}_idx"]).long() for s in ("train", "val", "test")}
        if overfit_n:
            self.idx["train"] = self.idx["train"][:overfit_n]
            self.idx["val"] = self.idx["train"]
        self.split = {s: (u0[i].to(device), ut[i].to(device)) for s, i in self.idx.items()}

    def norm(self, u):
        return (u - self.mean) / self.std

    def denorm(self, u):
        return u * self.std + self.mean


def autocast(device, bf16):
    if bf16 and device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


@torch.no_grad()
def predict(model, data, u0_phys, device, bf16, batch_size=1000, n_steps=None):
    """Physical-unit predictions. Recursive models return one tensor per refinement step."""
    model.eval()
    outs = []
    for s in range(0, len(u0_phys), batch_size):
        with autocast(device, bf16):
            if model.recursive:
                o = model(data.norm(u0_phys[s:s + batch_size]), n_steps=n_steps)
            else:
                o = [model(data.norm(u0_phys[s:s + batch_size]))]
        outs.append([data.denorm(p.float()) for p in o])
    return [torch.cat([b[k] for b in outs]) for k in range(len(outs[0]))]


def metrics(pred, true):
    """Mean per-trajectory relative L2 and global RMSE, both in physical units."""
    pred, true = pred.double(), true.double()
    rel = (pred - true).norm(dim=1) / true.norm(dim=1)
    return {"rel_l2_mean": rel.mean().item(), "rel_l2_median": rel.median().item(),
            "rel_l2_std": rel.std().item(), "rmse": (pred - true).pow(2).mean().sqrt().item()}


def per_traj_rel_l2(pred, true):
    return ((pred.double() - true.double()).norm(dim=1) / true.double().norm(dim=1)).cpu().numpy()

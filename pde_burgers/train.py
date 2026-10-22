"""Train one model (shallow | recursive | deep) on u(x,0) -> u(x,t*).

    python -m pde_burgers.train --model recursive --seed 0
    python -m pde_burgers.train --model recursive --overfit_n 64 --epochs 1500 --eval_every 100
    python -m pde_burgers.train --model recursive --bench --budget_minutes 210

Recursive training follows TRM deep supervision: for each of the K refinement steps the
encoder is re-run with the current weights, the step's three core calls are back-propagated,
the optimizer steps, and (y, z) are detached before the next step.
"""
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F

from models.ema import EMAHelper
from pde_burgers.common import BurgersData, autocast, load_config, metrics, predict
from pde_burgers.models import build_model, core_layer_calls, count_params

EXTRA_ARGS = [
    ("--model", dict(required=True, choices=["shallow", "recursive", "deep"])),
    ("--out_dir", dict(default=None)),
    ("--bench", dict(action="store_true", help="measure training throughput and exit")),
    ("--budget_minutes", dict(type=float, default=210.0)),
    ("--resume", dict(action="store_true")),
]


def lr_lambda(cfg, total_steps):
    def f(step):
        if step < cfg["warmup_steps"]:
            return (step + 1) / cfg["warmup_steps"]
        prog = min(1.0, (step - cfg["warmup_steps"]) / max(1, total_steps - cfg["warmup_steps"]))
        return 0.5 * (1 + math.cos(math.pi * prog))
    return f


class Trainer:
    def __init__(self, name, cfg, device, data):
        self.name, self.cfg, self.device, self.data = name, cfg, device, data
        self.model = build_model(name, cfg).to(device)
        self.ema = EMAHelper(mu=cfg["ema_decay"])
        self.ema.register(self.model)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        n_train = len(data.split["train"][0])
        self.batches_per_epoch = math.ceil(n_train / cfg["batch_size"])
        self.updates_per_batch = cfg["n_sup"] if self.model.recursive else 1
        self.total_steps = cfg["epochs"] * self.batches_per_epoch * self.updates_per_batch
        self.sched = torch.optim.lr_scheduler.LambdaLR(self.opt, lr_lambda(cfg, self.total_steps))
        self.step = 0

    def _update(self, loss):
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg["grad_clip"])
        self.opt.step()
        self.sched.step()
        self.ema.update(self.model)
        self.step += 1

    def train_batch(self, u0n, utn):
        """Returns the loss of every optimizer update made on this batch."""
        self.model.train()
        m, bf16 = self.model, self.cfg["bf16"]
        if not m.recursive:
            with autocast(self.device, bf16):
                pred = m(u0n)
            loss = F.mse_loss(pred.float(), utn)
            self._update(loss)
            return [loss.detach()]
        losses = []
        y, z = m.init_state(u0n)
        for _ in range(m.n_sup):
            with autocast(self.device, bf16):
                x = m.io.encode(u0n)  # recomputed after every optimizer update
                y, z = m.refine(x, y, z)
                pred = m.io.decode(y)
            loss = F.mse_loss(pred.float(), utn)
            self._update(loss)
            y, z = y.detach(), z.detach()
            losses.append(loss.detach())
        return losses

    def batches(self, gen):
        u0, ut = self.data.split["train"]
        perm = torch.randperm(len(u0), generator=gen).to(self.device)
        for s in range(0, len(u0), self.cfg["batch_size"]):
            i = perm[s:s + self.cfg["batch_size"]]
            yield self.data.norm(u0[i]), self.data.norm(ut[i])

    def validate(self):
        ema_model = self.ema.ema_copy(self.model)
        u0, ut = self.data.split["val"]
        preds = predict(ema_model, self.data, u0, self.device, self.cfg["bf16"])
        out = metrics(preds[-1], ut)
        if self.model.recursive:
            out["rel_l2_per_step"] = [metrics(p, ut)["rel_l2_mean"] for p in preds]
        return out, ema_model

    def state(self, epoch, best):
        return {"model": self.model.state_dict(), "ema": self.ema.state_dict(), "opt": self.opt.state_dict(),
                "sched": self.sched.state_dict(), "step": self.step, "epoch": epoch, "best": best}


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def bench(tr, cfg, budget_minutes, out_dir):
    gen = torch.Generator().manual_seed(0)
    it = iter(())
    def next_batch():
        nonlocal it
        try:
            return next(it)
        except StopIteration:
            it = tr.batches(gen)
            return next(it)
    for _ in range(10):
        tr.train_batch(*next_batch())
    sync(tr.device)
    n, t0 = 50, time.perf_counter()
    for _ in range(n):
        tr.train_batch(*next_batch())
    sync(tr.device)
    sec_per_batch = (time.perf_counter() - t0) / n
    t0 = time.perf_counter()
    tr.validate()
    sync(tr.device)
    val_sec = time.perf_counter() - t0
    sec_per_epoch = sec_per_batch * tr.batches_per_epoch
    per_epoch_with_val = sec_per_epoch + val_sec / cfg["eval_every"]
    res = {"model": tr.name, "batch_size": cfg["batch_size"], "sec_per_batch": sec_per_batch,
           "samples_per_sec": cfg["batch_size"] / sec_per_batch, "sec_per_epoch": sec_per_epoch,
           "val_sec": val_sec, "budget_minutes": budget_minutes,
           "epochs_fitting_budget": int(budget_minutes * 60 / per_epoch_with_val),
           "params": count_params(tr.model),
           "max_mem_gb": torch.cuda.max_memory_allocated(tr.device) / 1e9 if tr.device.type == "cuda" else None}
    print(json.dumps(res, indent=1))
    with open(os.path.join(out_dir, "bench.json"), "w") as f:
        json.dump(res, f, indent=1)


def main():
    cfg, extra = load_config(extra_args=EXTRA_ARGS)
    name = extra["model"]
    tag = f"overfit{cfg['overfit_n']}_" if cfg["overfit_n"] else ""
    out_dir = extra["out_dir"] or os.path.join("runs", f"{tag}{name}_s{cfg['seed']}")
    os.makedirs(out_dir, exist_ok=True)

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    data = BurgersData(cfg["data_cache"], device, cfg["overfit_n"])
    tr = Trainer(name, cfg, device, data)
    n_params = count_params(tr.model)
    print(f"[{name}] params={n_params:,} core layer calls/pred={core_layer_calls(name, cfg)} "
          f"train={len(data.split['train'][0])} val={len(data.split['val'][0])} device={device} "
          f"steps planned={tr.total_steps:,}")
    print(f"[data] input t={data.meta['input_time_actual']} target t={data.meta['target_time_actual']} "
          f"norm mean={data.mean:.5f} std={data.std:.5f}")

    if extra["bench"]:
        return bench(tr, cfg, extra["budget_minutes"], out_dir)

    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump({"model": name, **cfg, "params": n_params}, f, indent=1)

    start_epoch, best, train_sec = 0, {"rel_l2_mean": float("inf")}, 0.0
    last_path = os.path.join(out_dir, "last.pt")
    if extra["resume"] and os.path.exists(last_path):
        ck = torch.load(last_path, map_location=device)
        tr.model.load_state_dict(ck["model"])
        tr.ema.load_state_dict(ck["ema"])
        tr.opt.load_state_dict(ck["opt"])
        tr.sched.load_state_dict(ck["sched"])
        tr.step, start_epoch, best = ck["step"], ck["epoch"] + 1, ck["best"]
        train_sec = ck.get("train_sec", 0.0)
        print(f"[resume] from epoch {start_epoch}, step {tr.step}")

    gen = torch.Generator().manual_seed(cfg["seed"] * 1000 + start_epoch)
    log_f = open(os.path.join(out_dir, "log.jsonl"), "a")
    wall0 = time.perf_counter()
    epoch = start_epoch - 1
    for epoch in range(start_epoch, cfg["epochs"]):
        sync(device)
        t0 = time.perf_counter()
        step_losses = []
        for u0n, utn in tr.batches(gen):
            step_losses.append(torch.stack(tr.train_batch(u0n, utn)))
        sync(device)
        train_sec += time.perf_counter() - t0
        timed_out = train_sec / 60 > cfg["max_minutes"]
        last = epoch == cfg["epochs"] - 1 or timed_out

        if (epoch + 1) % cfg["eval_every"] == 0 or last:
            val, ema_model = tr.validate()
            rec = {"epoch": epoch, "step": tr.step, "lr": tr.sched.get_last_lr()[0],
                   "train_loss_per_update": torch.stack(step_losses).mean(0).tolist(),
                   "val_ema": val, "train_sec": train_sec, "wall_sec": time.perf_counter() - wall0}
            if val["rel_l2_mean"] < best["rel_l2_mean"]:
                best = {**val, "epoch": epoch, "step": tr.step}
                torch.save({"model_name": name, "cfg": cfg, "state_dict": ema_model.state_dict(),
                            "epoch": epoch, "step": tr.step, "val": val}, os.path.join(out_dir, "best_ema.pt"))
            rec["best_rel_l2"] = best["rel_l2_mean"]
            log_f.write(json.dumps(rec) + "\n")
            log_f.flush()
            per_step = f" per-step {[round(v, 5) for v in val['rel_l2_per_step']]}" if "rel_l2_per_step" in val else ""
            print(f"ep {epoch:5d} step {tr.step:7d} loss {rec['train_loss_per_update'][-1]:.3e} "
                  f"val relL2 {val['rel_l2_mean']:.5f} (best {best['rel_l2_mean']:.5f}@{best['epoch']})"
                  f"{per_step} train {train_sec/60:.1f}m", flush=True)
            torch.save({**tr.state(epoch, best), "train_sec": train_sec}, last_path)
        if timed_out:
            print(f"[{name}] time cap {cfg['max_minutes']} min reached at epoch {epoch}")
            break

    summary = {"model": name, "seed": cfg["seed"], "params": n_params, "epochs_done": epoch + 1,
               "epochs_planned": cfg["epochs"], "optimizer_steps": tr.step,
               "train_sec": train_sec, "best": best,
               "core_layer_calls_per_pred": core_layer_calls(name, cfg),
               "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"}
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()

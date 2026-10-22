# Does TRM-style recursion help on PDEBench 1D Burgers?

An 8-hour exploratory study on 2×A100 80GB. It asks whether TRM-style recursive refinement gives lower held-out
error than nonrecursive Transformers on the operator map u(x,0) → u(x,t*) for PDEBench 1D Burgers
(ν = 0.01, t* = stored time nearest 1.0), and what that costs in compute.

This builds on **Tiny Recursive Models** by Alexia Jolicoeur-Martineau (Samsung SAIL Montréal),
[paper](https://arxiv.org/abs/2510.04871), [original code](https://github.com/SamsungSAILMontreal/TinyRecursiveModels) (MIT).
Only `models/ema.py` is reused from that codebase; the recursion here is a simplified, continuous-output
re-implementation and is not the original TRM training recipe. Data: [PDEBench](https://github.com/pdebench/PDEBench)
(Takamoto et al., NeurIPS 2022 Datasets & Benchmarks).

## Setup

| | |
|---|---|
| Data | `1D_Burgers_Sols_Nu0.01.hdf5` only, with one (u(x,0), u(x,t*)) pair per trajectory and all 1024 points |
| Split | 80/10/10 trajectories, seed 42, made before preprocessing |
| Normalisation | one scalar mean/std fit on training inputs and targets pooled, then inverted before every metric |
| Patching | 128 patches × 8 values; `Linear(8,128)` encoder plus learned positional embeddings; `Linear(128,8)` decoder |
| Core | pre-norm, noncausal Transformer layers (width 128, 4 heads, FFN ×4, GELU) followed by a final LayerNorm |
| Recursive | learned y₀ and z₀ broadcast over patches. For each of K=4 steps: z←f(x+y+z) twice, then y←f(y+z), then û=D(y). MSE, backward, AdamW step, EMA update, detach (y,z). The encoder is re-run after each update. All calls share the same core weights. |
| Controls | (1) 2-layer, no recursion, about the same parameter count. (3) 24-layer untied, the same 24 layer applications per final prediction (4 × 3 × 2) and about 11× the parameters. |
| Optimiser | AdamW with lr 3e-4, wd 0.01 (all params), clip 1.0, batch 64, BF16 autocast, 200-step linear warmup then cosine decay to 0 |
| Selection | EMA weights (0.999 per optimizer update), validation mean relative L² checked every `eval_every` epochs. The test set is used only in `evaluate.py`. |

Every setting is in [`config.yaml`](config.yaml). Any key can be overridden as `--key value`.

## Exact commands (server)

```bash
git clone git@github.com:souspilot/TinyRecursiveBurger.git && cd TinyRecursiveBurger
conda env create -f pde_burgers/environment.yml   # Python 3.12, torch 2.7.1 (CUDA 12.6 wheels)
conda activate trm-burgers
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"   # expect: True 2
bash pde_burgers/run_server.sh data         # about 8 GB download, MD5 check, then logs/prepare_data.log
```
**Check `logs/prepare_data.log` before training.** It records the tensor shape, the `t-coordinate` length
(PDEBench 1D files store nt+1 times for nt snapshots), the chosen snapshot index and its actual time, and the
alternative time under the other indexing convention. The default, `--time_convention leading`, means snapshot k
is at t[k], so snapshot 0 is the initial condition.

```bash
bash pde_burgers/run_server.sh sanity       # overfit 64 trajectories (recursive on GPU0, shallow on GPU1)
bash pde_burgers/run_server.sh bench        # throughput; prints epochs_fitting_budget for each model
```
The sanity check passes when the overfit validation relL2, which is measured on those same 64 trajectories, falls
far below the persistence error and keeps falling. Next, set the epoch counts from `runs/bench_*/bench.json`.
Use `epochs_fitting_budget` with some margin. GPU1 runs shallow and then deep in sequence, so those two must
fit within about 4 hours together.

```bash
EPOCHS_RECURSIVE=<n> EPOCHS_SHALLOW=<n> EPOCHS_DEEP=<n> MAX_MIN=210 bash pde_burgers/run_server.sh train
tail -f logs/recursive_s0.log logs/shallow_s0.log logs/deep_s0.log
# optional, only if time remains:
EPOCHS_RECURSIVE=<n> EPOCHS_SHALLOW=<n> bash pde_burgers/run_server.sh train_seed1
# after all training has finished (needs an idle GPU for fair latency numbers):
bash pde_burgers/run_server.sh eval
```
To resume a crashed run, re-run its `python -m pde_burgers.train ...` command with `--resume`.

## Outputs

* `runs/<model>_s<seed>/`: `config.json`, `log.jsonl` (train loss for each update within a batch, EMA validation
  metrics, and validation error per refinement step for the recursive model), `best_ema.pt` (the selected
  checkpoint), `last.pt` (for resuming), `summary.json` (params, training time, optimizer steps)
* `results/metrics.json` and `results/results.md`: test mean per-trajectory relative L², physical-unit RMSE,
  parameters, core layer calls, latency at batch 1 and batch 1000, training time, reference baselines (persistence; training-mean field; ridge regression fit on train with alpha chosen on val), and the
  recursive model after steps 1–4. With multiple seeds, mean ± std is shown.
* `results/fields.png`: true and predicted fields plus |error| for test trajectories at the 10th, 50th and 90th
  error percentiles. `results/recursive_spatial_error_by_step.png`. `results/error_vs_refinement_and_latency.png`.

## Error by shock strength (follow-up analysis)

Recursion's RMSE advantage suggests it does better on high-amplitude, shock-dominated trajectories. To test
that, bin the test set by features of the true target (amplitude max−min, and max |∂u/∂x| as a measure of
shock strength) and compare each model's error per bin:

```bash
CUDA_VISIBLE_DEVICES=0 python -m pde_burgers.analyze_by_amplitude --runs 'runs/*_s*' 'runs_extra/shallow4x_s*' --out results_by_amplitude
```

This writes `results_by_amplitude/by_amplitude.md` and `results_by_amplitude/error_by_shock_strength.png`. The
markdown file has, for each bin: mean relative L², RMSE, each bin's share of the model's total squared error,
and the recursive model's error after each refinement step. Only finished runs (with `summary.json`) are used.
Runs are grouped by directory name without the `_s<seed>` suffix.

## Local smoke test (no GPU, synthetic data)

```bash
python -m pde_burgers.make_fake_data --out /tmp/smoke/fake.hdf5 --n 200
python -m pde_burgers.prepare_data --h5 /tmp/smoke/fake.hdf5 --out /tmp/smoke/cache.npz
python -m pde_burgers.train --model recursive --data_cache /tmp/smoke/cache.npz --epochs 2 --eval_every 1 --out_dir /tmp/smoke/runs/recursive_s0
python -m pde_burgers.evaluate --runs '/tmp/smoke/runs/*_s*' --data_cache /tmp/smoke/cache.npz --out /tmp/smoke/results --latency_batches 1 20
```
The synthetic data is advected and diffused Fourier modes, not Burgers. It only checks that the code runs.

## Interpreting results and limitations

* **Unequal training compute is part of the design.** With equal epochs, the recursive model makes 4 optimizer
  updates per batch and runs 12 core calls per batch. The shallow model makes 1 update and runs 1 core call. The
  deep model makes 1 update but runs 12× the shallow model's layers. Report training time and optimizer steps
  next to the errors (`results.md` does this). If the recursive model beats the shallow one but not the deep
  one, that points to extra compute rather than recursion as such.
* Do not claim that recursion helps unless it beats **both** controls by more than the seed-to-seed spread. With
  one seed per model, any difference is anecdotal.
* Only one target time, one viscosity and one dataset are tested. Refinement steps all predict the same t*, not
  successive times. There is no learned halting and no hyperparameter search: every model uses the same lr and
  schedule, which may suit some models better than others.
* Weight decay applies to every parameter, including LayerNorm, the positional embeddings and y₀/z₀.
* Latency is measured as the median of 50 BF16 forward passes, with normalisation included and the host-side
  denormalisation excluded, on the GPU named in `metrics.json`. Batch-1 latency is dominated by kernel launch
  overhead, so it scales with the number of layer calls rather than with FLOPs. If other jobs share the GPU,
  latency and training time are inflated.
* The `t-coordinate` convention (see above) is an assumption, and it is logged. If the other convention is right,
  t* is 1.01 rather than 1.00.

# TinyRecursiveBurger: does "thinking longer" help a tiny network solve a PDE?

An 8-hour study of **TRM-style recursive refinement** ([Tiny Recursive Models](https://arxiv.org/abs/2510.04871))
on [PDEBench](https://arxiv.org/abs/2210.07182) **1D Burgers' equation** (ν = 0.01). The task is to predict the
field at t = 1, after shocks have formed, from the initial condition.

The recursive model is a 416K-parameter Transformer that reuses its 2-layer core 12 times. It is compared against
controls for **the same size**, **the same compute** and **the same number of training updates**, with 2 seeds each:

| Model | Params | MFLOPs/sample | Test rel. L² | Test RMSE |
|---|---:|---:|---:|---:|
| Ridge regression (best linear map) | 1.05M | 2 | 0.343 | 0.134 |
| 2-layer Transformer | 415K | 118 | 0.0172 | 0.0113 |
| 2-layer, trained for the same number of updates as recursive | 415K | 118 | **0.0112** | 0.0108 |
| 24-layer untied Transformer (same compute) | 4.78M | 1,410 | 0.0131 | 0.0129 |
| Recursive, 4 refinement steps | 416K | 1,410 | 0.0135 | **0.0073** |
| Recursive, adaptive halting (no retraining) | 416K | 815 | 0.0126 | **0.0073** |

**Findings.** The average hides a regime split. With the same number of training updates, a plain 2-layer network
is the most accurate model overall. But:

* The recursive model's error is **nearly independent of shock strength** (×1.08 from the smoothest to the
  most-shocked quarter, against ×1.6–2.0 for every other model). That makes it the most accurate model on the
  strongest shocks and one of the least accurate on smooth waves.
* **Refinement helps on shocks** (−12% from step 1 to step 3) **and hurts on smooth inputs** (+18% by step 4).
* **The model's first guess tells it how long to think.** Choosing the number of steps from the sharpness of the
  step-1 prediction (no retraining) matches the 24-layer model with 11× fewer parameters and 58% of its compute.
  A hindsight oracle over the 4 intermediate predictions reaches the best model's accuracy, so the bottleneck is
  deciding when to stop.

**Read more**

* **Write-up:** [`pde_burgers/WRITEUP.md`](pde_burgers/WRITEUP.md)
* **Exact commands and limitations:** [`pde_burgers/README.md`](pde_burgers/README.md)

## Quick start

You need a GPU and an 8 GB download.

```bash
conda env create -f pde_burgers/environment.yml && conda activate trm-burgers
bash pde_burgers/run_server.sh data      # download PDEBench 1D Burgers (MD5-checked) and preprocess
EPOCHS_RECURSIVE=1000 EPOCHS_SHALLOW=1000 EPOCHS_DEEP=1000 bash pde_burgers/run_server.sh train   # 2 GPUs, ~2.5 h
bash pde_burgers/run_server.sh eval      # test metrics, latency, figures -> results/
python -m pde_burgers.analyze_by_amplitude --runs 'runs/*_s*' 'runs_extra/shallow4x_s*'   # error by shock strength
python -m pde_burgers.adaptive_halting --runs 'runs/recursive_s*'                          # inference-time halting
```

Trained checkpoints (`best_ema.pt`, selected on validation) for every run are in `runs/` and `runs_extra/`. You can
evaluate them without retraining after running the `data` step.

## Credits

* **Tiny Recursive Models:** this repository builds on the original code by **Alexia Jolicoeur-Martineau**
  (Samsung SAIL Montréal), [upstream repository](https://github.com/SamsungSAILMontreal/TinyRecursiveModels),
  MIT License (see `LICENSE`). The recursion studied here is a simplified, continuous-output re-implementation
  in `pde_burgers/`.
* **The original TRM code is still here.** `pretrain.py`, `models/`, `dataset/`, `config/` and the rest are
  unchanged. For the original ARC-AGI, Sudoku and Maze experiments and their commands, see the
  [original README](https://github.com/SamsungSAILMontreal/TinyRecursiveModels/blob/main/README.md).
* **Data:** PDEBench (Takamoto et al., NeurIPS 2022), [doi:10.18419/darus-2986](https://doi.org/10.18419/darus-2986).

If you use the recursive model, please cite the original TRM paper:

```bibtex
@misc{jolicoeurmartineau2025morerecursivereasoningtiny,
      title={Less is More: Recursive Reasoning with Tiny Networks},
      author={Alexia Jolicoeur-Martineau},
      year={2025},
      eprint={2510.04871},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2510.04871},
}
```

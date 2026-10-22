Test set: n=1000 trajectories, target t=1.0 (snapshot 100), device NVIDIA A100-SXM4-80GB, bf16=True.
Mean ± sample std over seeds when more than one seed exists.

| model | seeds | params | core layer calls | MFLOPs/sample | val relL2 | test relL2 (mean) | test RMSE | latency bs=1 (ms) | latency bs=1000 (ms) | train time (min) | optimizer steps |
|---|---|---|---|---|---|---|---|---|---|---|---|
| training-mean field | - | 1,024 | 0 | 0.0 | 0.99987 | 1.00470 | 0.60261 | - | - | - | - |
| persistence u(x,0) | - | 0 | 0 | 0.0 | - | 1.76686 | 0.52373 | - | - | - | - |
| ridge regression (linear) | - | 1,049,600 | 0 | 2.1 | 0.35924 | 0.34342 | 0.13354 | - | - | - | - |
| 2-layer (no recursion) | 2 | 415,368 | 2 | 118.0 | 0.01736 ± 0.00283 | 0.01717 ± 0.00260 | 0.01125 ± 0.00188 | 1.15 ± 0.01 | 4.36 ± 0.00 | 15.4 ± 2.2 | 125000 ± 0 |
| recursive TRM (2-layer core) | 2 | 415,624 | 24 | 1,409.8 | 0.01312 ± 0.00082 | 0.01351 ± 0.00065 | 0.00729 ± 0.00033 | 8.66 ± 0.04 | 50.02 ± 0.02 | 128.9 ± 17.9 | 500000 ± 0 |
| 24-layer untied | 2 | 4,777,352 | 24 | 1,409.8 | 0.01309 ± 0.00062 | 0.01309 ± 0.00064 | 0.01288 ± 0.00285 | 10.41 ± 0.07 | 44.78 ± 0.00 | 115.0 ± 3.6 | 125000 ± 0 |

Recursive model after each refinement step (test, same checkpoint):

| step | core layer calls | MFLOPs/sample | test relL2 | test RMSE | latency bs=1 (ms) | latency bs=1000 (ms) |
|---|---|---|---|---|---|---|
| 1 | 6 | 352.8 | 0.01327 ± 0.00050 | 0.00839 ± 0.00015 | 2.54 ± 0.00 | 12.68 ± 0.00 |
| 2 | 12 | 705.2 | 0.01294 ± 0.00082 | 0.00760 ± 0.00023 | 4.57 ± 0.01 | 25.15 ± 0.03 |
| 3 | 18 | 1,057.5 | 0.01275 ± 0.00067 | 0.00726 ± 0.00030 | 6.62 ± 0.01 | 37.60 ± 0.05 |
| 4 | 24 | 1,409.8 | 0.01351 ± 0.00065 | 0.00729 ± 0.00033 | 8.66 ± 0.04 | 50.02 ± 0.02 |

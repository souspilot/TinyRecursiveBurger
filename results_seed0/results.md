Test set: n=1000 trajectories, target t=1.0 (snapshot 100), device NVIDIA A100-SXM4-80GB, bf16=True.
Mean ± sample std over seeds when more than one seed exists.

| model | seeds | params | core layer calls | val relL2 | test relL2 (mean) | test RMSE | latency bs=1 (ms) | latency bs=1000 (ms) | train time (min) | optimizer steps |
|---|---|---|---|---|---|---|---|---|---|---|
| training-mean field | - | 1,024 | 0 | 0.99987 | 1.00470 | 0.60261 | - | - | - | - |
| persistence u(x,0) | - | 0 | 0 | - | 1.76686 | 0.52373 | - | - | - | - |
| ridge regression (linear) | - | 1,049,600 | 0 | 0.35924 | 0.34342 | 0.13354 | - | - | - | - |
| 2-layer (no recursion) | 1 | 415,368 | 2 | 0.01537 | 0.01533 | 0.00992 | 1.14 | 4.36 | 17.0 | 125000 |
| recursive TRM (2-layer core) | 1 | 415,624 | 24 | 0.01370 | 0.01397 | 0.00752 | 8.66 | 49.94 | 141.5 | 500000 |
| 24-layer untied | 1 | 4,777,352 | 24 | 0.01265 | 0.01263 | 0.01490 | 10.38 | 44.70 | 112.5 | 125000 |

Recursive model after each refinement step (test, same checkpoint):

| step | core layer calls | test relL2 | test RMSE | latency bs=1 (ms) | latency bs=1000 (ms) |
|---|---|---|---|---|---|
| 1 | 6 | 0.01363 | 0.00849 | 2.51 | 12.67 |
| 2 | 12 | 0.01352 | 0.00777 | 4.57 | 25.09 |
| 3 | 18 | 0.01322 | 0.00747 | 6.62 | 37.51 |
| 4 | 24 | 0.01397 | 0.00752 | 8.66 | 49.94 |

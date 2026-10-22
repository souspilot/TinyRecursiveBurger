Test set: n=1000 trajectories, target t=1.0 (snapshot 100), device NVIDIA A100-SXM4-80GB, bf16=True.
Mean ± sample std over seeds when more than one seed exists.

| model | seeds | params | core layer calls | MFLOPs/sample | val relL2 | test relL2 (mean) | test RMSE | latency bs=1 (ms) | latency bs=1000 (ms) | train time (min) | optimizer steps |
|---|---|---|---|---|---|---|---|---|---|---|---|
| training-mean field | - | 1,024 | 0 | 0.0 | 0.99987 | 1.00470 | 0.60261 | - | - | - | - |
| persistence u(x,0) | - | 0 | 0 | 0.0 | - | 1.76686 | 0.52373 | - | - | - | - |
| ridge regression (linear) | - | 1,049,600 | 0 | 2.1 | 0.35924 | 0.34342 | 0.13354 | - | - | - | - |
| 2-layer (no recursion) | 2 | 415,368 | 2 | 118.0 | 0.01157 ± 0.00186 | 0.01117 ± 0.00170 | 0.01080 ± 0.00215 | 1.16 ± 0.00 | 4.62 ± 0.37 | 53.0 ± 0.2 | 500000 ± 0 |

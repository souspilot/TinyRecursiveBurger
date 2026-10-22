# Adaptive halting (inference-time, no retraining)

Recursive model, 2 seed(s), mean ± std. Policies were fitted on validation (4 equal-count bins of the difficulty score; best step per bin) and applied once to test.

| output | test rel L2 | test RMSE | mean steps | mean MFLOPs/sample |
|---|---:|---:|---:|---:|
| fixed step 1 | 0.0133 ± 0.0005 | 0.0084 ± 0.0001 | 1.00 ± 0.00 | 353 ± 0 |
| fixed step 2 | 0.0129 ± 0.0008 | 0.0076 ± 0.0002 | 2.00 ± 0.00 | 705 ± 0 |
| fixed step 3 | 0.0127 ± 0.0007 | 0.0073 ± 0.0003 | 3.00 ± 0.00 | 1057 ± 0 |
| fixed step 4 (reported in main results) | 0.0135 ± 0.0006 | 0.0073 ± 0.0003 | 4.00 ± 0.00 | 1410 ± 0 |
| **adaptive**, score = max |∂û/∂x| of the step-1 prediction | 0.0126 ± 0.0007 | 0.0073 ± 0.0003 | 2.31 ± 0.21 | 815 ± 74 |
| **adaptive**, score = max |∂u/∂x| of the input u(x,0) | 0.0127 ± 0.0006 | 0.0074 ± 0.0004 | 2.58 ± 0.18 | 911 ± 64 |
| oracle: best step per trajectory, chosen with the test target (upper bound) | 0.0114 ± 0.0007 | 0.0067 ± 0.0003 | 2.18 ± 0.08 | 769 ± 29 |

Steps chosen per validation bin (weakest → strongest score), per seed:

* `runs/recursive_s0`: pred_grad: [1, 3, 2, 3]; u0_grad: [2, 2, 3, 3]
* `runs/recursive_s1`: pred_grad: [1, 3, 3, 3]; u0_grad: [2, 3, 3, 3]

Reference (final output, test rel L2): 2-layer 0.0172, 24-layer untied 0.0131, 2-layer, 4x epochs 0.0112

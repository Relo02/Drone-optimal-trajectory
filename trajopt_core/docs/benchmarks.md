# Formulation benchmarks

Scenario: `corridor_with_pillars` — 4 m corridor, three staggered pillars, no prior map (6 m LiDAR).  Closed loop matches deployment: the MPC publishes a lookahead setpoint tracked by the platform inner loop.


### 1. Does build-once pay? Three obstacle-term strategies (legged, corridor)

| variant | n_builds | build_ms_total | solve_ms_p50 | solve_ms_p95 | total_ms_mean | iter_mean |
|---|---|---|---|---|---|---|
| point barrier, parametric | 1 | 270.431 | 29.256 | 36.573 | 121.892 | 12.600 |
| grid B-spline, baked (rebuild) | 15 | 1945.102 | 50.386 | 82.041 | 190.120 | 8.133 |
| grid B-spline, parametric | 1 | 213.718 | 14325.174 | 23031.951 | 15236.983 | 8.133 |

### 2. NLP size and sparsity vs horizon (multiple shooting)

| platform | N | n_variables | n_constraints | jac_nnz | jac_density | hess_density |
|---|---|---|---|---|---|---|
| aerial | 10 | 117 | 137 | 317 | 0.020 | 0.056 |
| aerial | 20 | 227 | 267 | 627 | 0.010 | 0.029 |
| aerial | 30 | 337 | 397 | 937 | 0.007 | 0.020 |
| aerial | 50 | 557 | 657 | 1557 | 0.004 | 0.012 |
| legged | 10 | 63 | 63 | 183 | 0.046 | 0.076 |
| legged | 20 | 123 | 123 | 363 | 0.024 | 0.040 |
| legged | 30 | 183 | 183 | 543 | 0.016 | 0.027 |
| legged | 50 | 303 | 303 | 903 | 0.010 | 0.016 |

### 3. Obstacle penalty smoothness: C^1 hinge^2 vs C^inf logistic barrier

| variant | reached | success_rate | iter_mean | iter_max | solve_ms_mean | solve_ms_p95 | clr_min_m |
|---|---|---|---|---|---|---|---|
| aerial / hinge^2 (C1) | True | 1.000 | 5.174 | 11 | 23.699 | 38.753 | 0.723 |
| aerial / logistic (Cinf) | True | 1.000 | 14.417 | 22 | 70.321 | 130.145 | 0.862 |
| legged / hinge^2 (C1) | True | 1.000 | 5.383 | 12 | 12.874 | 22.269 | 0.720 |
| legged / logistic (Cinf) | True | 1.000 | 20.258 | 51 | 47.231 | 84.677 | 0.745 |

### 4. Exact (L1) vs quadratic (L2) slack penalty

| rho | max_slack_l1 | max_slack_l2 | iter_l1 | iter_l2 |
|---|---|---|---|---|
| 20.000 | 0.135 | 0.179 | 13 | 16 |
| 200.000 | 0.000 | 0.107 | 13 | 16 |
| 2000.000 | 0.000 | 0.025 | 14 | 16 |
| 20000.000 | 0.000 | 0.003 | 16 | 17 |

### 5a. SE(2) local truncation error vs step size

| dt_s | err_euler_m | err_midpoint_m |
|---|---|---|
| 0.200 | 0.020 | 0.000 |
| 0.100 | 0.005 | 0.000 |
| 0.050 | 0.001 | 0.000 |
| 0.025 | 0.000 | 0.000 |
| fitted order | 1.999 | 3.000 |

### 5b. Closed-loop effect (legged, corridor)

| variant | reached | time_s | len_m | clr_min_m | err_mean_m | iter_mean | solve_ms_mean |
|---|---|---|---|---|---|---|---|
| legged / euler | True | 17.400 | 15.326 | 0.749 | 0.022 | 19.080 | 45.086 |
| legged / midpoint | True | 17.100 | 15.292 | 0.727 | 0.019 | 18.737 | 46.188 |

### 6. Initial guess: shifted solution vs reference vs cold start

| variant | iter_mean | iter_max | solve_ms_mean | success_rate | reached |
|---|---|---|---|---|---|
| open_field             / aerial / shifted previous solution | 5.241 | 11 | 21.362 | 1.000 | True |
| open_field             / aerial / reference trajectory | 4.330 | 11 | 17.552 | 1.000 | True |
| open_field             / aerial / cold (zeros) | 6.375 | 13 | 29.778 | 1.000 | True |
| open_field             / legged / shifted previous solution | 5.000 | 6 | 17.369 | 1.000 | True |
| open_field             / legged / reference trajectory | 4.000 | 4 | 12.715 | 1.000 | True |
| open_field             / legged / cold (zeros) | 5.000 | 5 | 14.983 | 1.000 | True |
| corridor_with_pillars  / aerial / shifted previous solution | 14.972 | 24 | 66.601 | 1.000 | True |
| corridor_with_pillars  / aerial / reference trajectory | 13.140 | 22 | 70.758 | 1.000 | True |
| corridor_with_pillars  / aerial / cold (zeros) | 13.184 | 20 | 62.318 | 1.000 | True |
| corridor_with_pillars  / legged / shifted previous solution | 18.737 | 42 | 54.370 | 1.000 | True |
| corridor_with_pillars  / legged / reference trajectory | 17.825 | 46 | 55.508 | 1.000 | True |
| corridor_with_pillars  / legged / cold (zeros) | 16.257 | 35 | 44.571 | 1.000 | True |

### 7. A* polyline conditioning, both closed-loop modes (150-cycle budget)

| variant | goal_dist_m | len_m | err_mean_m | clr_min_m | iter_mean | solve_ms_mean |
|---|---|---|---|---|---|---|
| lookahead / aerial / raw | 2.910 | 10.887 | 0.007 | 0.886 | 16.260 | 75.894 |
| lookahead / aerial / conditioned | 2.731 | 11.007 | 0.008 | 0.878 | 16.033 | 65.756 |
| lookahead / legged / raw | 2.242 | 13.590 | 0.023 | 0.796 | 20.473 | 44.250 |
| lookahead / legged / conditioned | 1.868 | 13.741 | 0.020 | 0.727 | 19.593 | 39.382 |
| direct    / aerial / raw | 8.923 | 3.997 | 0.004 | 0.942 | 16.507 | 63.370 |
| direct    / aerial / conditioned | 7.534 | 5.528 | 0.005 | 0.944 | 16.927 | 64.469 |
| direct    / legged / raw | 11.296 | 8.497 | 0.026 | 1.897 | 14.647 | 29.327 |
| direct    / legged / conditioned | 11.284 | 8.811 | 0.027 | 1.888 | 14.960 | 31.874 |

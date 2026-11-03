Effective sweep width of the env sensor (measured): 57.2 m (nominal POD x width = 34.5 m)

## 1. Does the approximation rank rectangles like the simulation?

47 decision points, 21 candidate rectangles each (the planned rectangles + random shifts/resizes/rotations).

| POC formula | Spearman rank corr. (mean, median) | sim POS of approx-best / sim-best |
|---|---|---|
| SAROPS: snapshot at cycle start, no transit | 0.62, 0.71 | 87% |
| time average over search window, L minus transit | 0.80, 0.87 | 85% |

sim / approx (tavg) median 2.62, IQR 1.36-5.78

## 2. Rectangle placed by each variant: simulated POS of the next 10 min

| variant / cell (m) | sim POS per cycle | relative to tavg_pca/100 | approx / sim | mean size (km) |
|---|---|---|---|---|
| snap0/100 | 0.0259 | 87% | 1.02 | 0.6 x 0.4 |
| mid/100 | 0.0285 | 96% | 0.89 | 0.6 x 0.3 |
| tavg/100 | 0.0281 | 95% | 0.82 | 0.6 x 0.3 |
| tavg_pca/100 | 0.0296 | 100% | 0.90 | 0.7 x 0.2 |
| tavg_pca/200 | 0.0296 | 100% | 0.82 | 0.8 x 0.3 |

## 3. Whole episode (one rectangle per 10-min cycle, no simulation in the planner)

| method | POS mean ± sd | min |
|---|---|---|
| v2 sector T10 (compare_patterns_v2) | 0.188 ± 0.088 | 0.093 |
| v2 parallel T10 (compare_patterns_v2) | 0.181 ± 0.088 | 0.085 |
| v2 mixed T10 (compare_patterns_v2) | 0.181 ± 0.086 | 0.081 |
| tavg_pca/200 | 0.179 ± 0.089 | 0.087 |
| tavg_pca/100 (with eval) | 0.178 ± 0.090 | 0.081 |
| tavg_pca/100 | 0.178 ± 0.090 | 0.081 |
| greedy90 (compare_patterns_v2) | 0.170 ± 0.117 | 0.000 |
| mid/100 | 0.168 ± 0.088 | 0.082 |
| tavg/100 | 0.165 ± 0.091 | 0.079 |
| snap0/100 | 0.164 ± 0.076 | 0.083 |

Scenarios: [300, 301, 302, 303, 304, 305, 306, 307].
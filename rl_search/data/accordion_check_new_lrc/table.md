Sensor: new (['search.fov_cross_m=286', 'search.fov_along_m=207', 'search.pod=0.7']), detection model: lrc

Effective sweep width of the env sensor (measured): 203.0 m (nominal POD x width = 0.7 x 286 m = 200 m)

## 1. Does the approximation rank rectangles like the simulation?

47 decision points, 21 candidate rectangles each (the planned rectangles + random shifts/resizes/rotations).

| POC formula | Spearman rank corr. (mean, median) | sim POS of approx-best / sim-best |
|---|---|---|
| SAROPS: snapshot at cycle start, no transit | 0.46, 0.49 | 82% |
| time average over search window, L minus transit | 0.68, 0.76 | 86% |

sim / approx (tavg) median 2.53, IQR 1.65-4.79

## 2. Rectangle placed by each variant: simulated POS of the next 10 min

| variant / cell (m) | sim POS per cycle | relative to tavg_pca/100 | approx / sim | mean size (km) |
|---|---|---|---|---|
| snap0/100 | 0.0466 | 83% | 0.88 | 0.9 x 0.8 |
| mid/100 | 0.0514 | 92% | 0.75 | 0.8 x 0.8 |
| tavg/100 | 0.0516 | 92% | 0.65 | 0.8 x 0.6 |
| tavg_pca/100 | 0.0560 | 100% | 0.70 | 1.0 x 0.5 |
| tavg_pca/200 | 0.0602 | 107% | 0.68 | 1.2 x 0.5 |

## 3. Whole episode (one rectangle per 10-min cycle, no simulation in the planner)

| method | POS mean ± sd | min |
|---|---|---|
| tavg_pca/200 | 0.345 ± 0.118 | 0.200 |
| tavg_pca/100 (with eval) | 0.336 ± 0.105 | 0.206 |
| tavg_pca/100 | 0.336 ± 0.105 | 0.206 |
| tavg/100 | 0.328 ± 0.128 | 0.174 |
| mid/100 | 0.322 ± 0.110 | 0.195 |
| snap0/100 | 0.320 ± 0.100 | 0.196 |

Scenarios: [300, 301, 302, 303, 304, 305, 306, 307].
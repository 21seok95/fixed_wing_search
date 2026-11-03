scenario 301, sweep width 203 m, decision-point checks 8 s

## Rectangle each formula picks (same state, accordion only)

| t (min) | formula | size (km) | spacing (m) | approx POS | simulated gain |
|---|---|---|---|---|---|
| 2 | exp | 1.4 x 0.4 | 200 | 0.054 | 0.013 |
| 2 | lanes | 1.4 x 0.6 | 300 | 0.068 | 0.013 |
| 12 | exp | 1.4 x 0.4 | 133 | 0.181 | 0.245 |
| 12 | lanes | 1.6 x 0.4 | 133 | 0.202 | 0.243 |
| 22 | exp | 1.2 x 0.6 | 150 | 0.043 | 0.064 |
| 22 | lanes | 3.4 x 0.4 | 400 | 0.051 | 0.061 |
| 32 | exp | 1.0 x 0.4 | 200 | 0.044 | 0.066 |
| 32 | lanes | 2.8 x 0.4 | 400 | 0.065 | 0.082 |
| 42 | exp | 3.0 x 0.4 | 200 | 0.032 | 0.048 |
| 42 | lanes | 3.4 x 0.4 | 400 | 0.039 | 0.050 |
| 52 | exp | 1.0 x 0.8 | 267 | 0.029 | 0.046 |
| 52 | lanes | 1.2 x 0.8 | 400 | 0.038 | 0.045 |

exp: mean spacing 192 m, mean simulated gain 0.0804, approx/sim 0.67, rank corr. with simulation 0.57 (6 points x 18 rectangles)

lanes: mean spacing 339 m, mean simulated gain 0.0824, approx/sim 0.83, rank corr. with simulation 0.57 (6 points x 18 rectangles)

## Whole episode

| planner | formula | POS (planner's particles) | held-out POS | s |
|---|---|---|---|---|
| accordion | exp | 0.483 | 0.476 | 1 |
| accordion | lanes | 0.464 | 0.462 | 1 |
| SAROPS | exp | 0.487 | 0.483 | 36 |
| SAROPS | lanes | 0.498 | 0.487 | 36 |
exit 0

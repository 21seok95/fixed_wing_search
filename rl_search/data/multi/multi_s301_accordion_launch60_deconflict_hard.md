N=1: POS 0.470, held-out 0.461, plans 4, 2 s
N=2: POS 0.581, held-out 0.567, plans 9, 13 s
N=3: POS 0.644, held-out 0.638, plans 13, 21 s
N=4: POS 0.699, held-out 0.696, plans 17, 34 s
N=5: POS 0.738, held-out 0.736, plans 23, 37 s

scenario 301, planner accordion, task mode (search 10 min, transit <= 10 min, lanes cover), launch 60 s apart from the launch point, separation 300 m as a constraint

| UAVs | POS | held-out POS | gain over 1 UAV | POS at 15 / 30 / 45 min | plans | closest pair (m) | time any pair < 300 m (s) | of it in the first 15 min | encounters < 300 m | plans with holds | plans without a feasible option | s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.470 | 0.461 | 1.00x | 0.086 / 0.319 / 0.406 | 4 | inf | 0 | 0 | 0 | 0 | 0 | 2 |
| 2 | 0.581 | 0.567 | 1.23x | 0.189 / 0.408 / 0.527 | 9 | 408 | 0 | 0 | 0 | 2 | 0 | 13 |
| 3 | 0.644 | 0.638 | 1.38x | 0.280 / 0.504 / 0.592 | 13 | 289 | 5 | 0 | 1 | 2 | 1 | 21 |
| 4 | 0.699 | 0.696 | 1.51x | 0.288 / 0.542 / 0.645 | 17 | 211 | 25 | 0 | 2 | 2 | 2 | 34 |
| 5 | 0.738 | 0.736 | 1.60x | 0.312 / 0.573 / 0.696 | 23 | 289 | 5 | 0 | 1 | 3 | 1 | 37 |
rl_search/data/multi/multi_s301_accordion_launch60_deconflict_hard.png

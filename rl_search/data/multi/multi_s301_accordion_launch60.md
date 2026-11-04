N=1: POS 0.470, held-out 0.461, plans 4, 2 s
N=2: POS 0.594, held-out 0.588, plans 9, 4 s
N=3: POS 0.649, held-out 0.641, plans 15, 9 s
N=4: POS 0.687, held-out 0.687, plans 18, 11 s
N=5: POS 0.725, held-out 0.722, plans 22, 17 s

scenario 301, planner accordion, task mode (search 10 min, transit <= 10 min, lanes cover), launch 60 s apart from the launch point

| UAVs | POS | held-out POS | gain over 1 UAV | POS at 15 / 30 / 45 min | plans | closest pair (m) | time any pair < 300 m (s) | of it in the first 15 min | encounters < 300 m | plans with holds | plans without a feasible option | s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.470 | 0.461 | 1.00x | 0.086 / 0.319 / 0.406 | 4 | inf | 0 | 0 | 0 | 0 | 0 | 2 |
| 2 | 0.594 | 0.588 | 1.28x | 0.200 / 0.432 / 0.541 | 9 | 289 | 5 | 0 | 1 | 0 | 0 | 4 |
| 3 | 0.649 | 0.641 | 1.39x | 0.280 / 0.498 / 0.606 | 15 | 111 | 25 | 0 | 2 | 0 | 0 | 9 |
| 4 | 0.687 | 0.687 | 1.49x | 0.307 / 0.557 / 0.649 | 18 | 99 | 130 | 30 | 9 | 0 | 0 | 11 |
| 5 | 0.725 | 0.722 | 1.57x | 0.331 / 0.586 / 0.675 | 22 | 32 | 95 | 30 | 6 | 0 | 0 | 17 |

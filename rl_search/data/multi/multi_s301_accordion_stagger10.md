N=1: POS 0.470, held-out 0.461, plans 4, 2 s
N=2: POS 0.546, held-out 0.536, plans 10, 4 s
N=3: POS 0.646, held-out 0.653, plans 14, 7 s
N=4: POS 0.703, held-out 0.701, plans 17, 11 s
N=5: POS 0.729, held-out 0.730, plans 23, 16 s

scenario 301, planner accordion, task mode (search 10 min, transit <= 10 min, lanes cover), launch 10 s apart

| UAVs | POS | held-out POS | gain over 1 UAV | POS at 15 / 30 / 45 min | plans | closest pair (m) | time any pair < 300 m (s) | encounters < 300 m | s |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.470 | 0.461 | 1.00x | 0.086 / 0.318 / 0.406 | 4 | inf | 0 | 0 | 2 |
| 2 | 0.546 | 0.536 | 1.16x | 0.166 / 0.395 / 0.496 | 10 | 100 | 805 | 7 | 4 |
| 3 | 0.646 | 0.653 | 1.42x | 0.267 / 0.496 / 0.608 | 14 | 15 | 1500 | 16 | 7 |
| 4 | 0.703 | 0.701 | 1.52x | 0.315 / 0.551 / 0.660 | 17 | 15 | 1675 | 17 | 11 |
| 5 | 0.729 | 0.730 | 1.59x | 0.335 / 0.596 / 0.689 | 23 | 15 | 2115 | 21 | 16 |
rl_search/data/multi/multi_s301_accordion_stagger10.png

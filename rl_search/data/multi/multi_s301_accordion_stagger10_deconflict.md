N=2: POS 0.546, held-out 0.536, plans 10, 8 s
N=3: POS 0.649, held-out 0.652, plans 14, 12 s
N=5: POS 0.734, held-out 0.733, plans 23, 22 s

scenario 301, planner accordion, task mode (search 10 min, transit <= 10 min, lanes cover), launch 10 s apart, planning-time separation 300 m

| UAVs | POS | held-out POS | gain over 1 UAV | POS at 15 / 30 / 45 min | plans | closest pair (m) | time any pair < 300 m (s) | of it in the first 15 min | encounters < 300 m | s |
|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 0.546 | 0.536 | - | 0.166 / 0.395 / 0.496 | 10 | 100 | 805 | 695 | 7 | 8 |
| 3 | 0.649 | 0.652 | - | 0.267 / 0.495 / 0.597 | 14 | 15 | 1465 | 1255 | 14 | 12 |
| 5 | 0.734 | 0.733 | - | 0.335 / 0.580 / 0.695 | 23 | 15 | 1960 | 1725 | 19 | 22 |
rl_search/data/multi/multi_s301_accordion_stagger10_deconflict.png

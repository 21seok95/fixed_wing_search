Fix (intercept_adjacent, setting no_overlap) on and off, SAROPS, scenarios 500-509. A at x 70 for every row, so rows compare. Differences are on minus off, paired by scenario.

| UAVs | config | POS off / on | POS diff (se, higher in) | A x 70 off / on | A diff (se) | pair < 300 m off / on (s) | closest pair on (min, m) | close time on, 0-15 / 15-30 / 30-45 / 45-60 / after (s) | holds per flight (min of holding) | plans with no feasible option | s/episode on |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 5 | A + 1 POS, x 70 | 0.707 / 0.706 | -0.001 (0.011, 5/10) | 0.316 / 0.334 | +0.018 (0.008) | 42 / 11 | 6 | 6 / 3 / 2 / 0 / 0 | 1.6 (2.7) | 5 | 225 |

Plans with no rectangle clear of the others (normal choice flown), of all plans: 5 UAVs 41 of 219

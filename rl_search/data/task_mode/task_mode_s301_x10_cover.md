
accordion, task, lanes k=1-3: tasks (plan min, transit est. / actual min, length min, end)
    2.0   8.9 /  8.7   20.1  done  1.4x1.7 km k=1
   22.1   0.7 /  0.6    8.9  done  0.6x1.1 km k=2
   31.0   6.6 /  6.8   16.3  done  1.2x0.6 km k=3
   47.3   5.7 /  5.5   12.5  episode end  0.6x1.1 km k=2

SAROPS, task, lanes k=1-3: tasks (plan min, transit est. / actual min, length min, end)
    2.0   9.7 /  9.5   20.9  done  1.4x1.7 km k=1
   22.9   0.1 /  0.1    8.8  done  0.6x0.9 km k=3
   31.8   5.8 /  5.6   15.2  done  1.2x0.6 km k=3
   47.0   5.2 /  5.0   12.8  episode end  1.2x0.9 km k=2

scenario 301, sweep width 203 m, search 10 min, max transit 10 min

| planner | POS | held-out POS | POS at 15 / 30 min | time not on lanes | first lane (min after handover) | plans | jumps > 3 km | s |
|---|---|---|---|---|---|---|---|---|
| accordion, task, lanes k=1-3 | 0.470 | 0.461 | 0.086 / 0.318 | 37% | 8.6 | 4 | 2 | 1 |
| SAROPS, task, lanes k=1-3 | 0.490 | 0.480 | 0.122 / 0.354 | 35% | 9.4 | 4 | 2 | 37 |
rl_search/data/task_mode/task_mode_s301_x10_cover.png

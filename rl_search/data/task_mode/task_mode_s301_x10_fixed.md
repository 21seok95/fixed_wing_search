
accordion, task, lanes no overlap: tasks (plan min, transit est. / actual min, length min, end)
    2.0   8.7 /  8.5   19.9  done
   21.9   0.4 /  0.6   12.1  done
   34.0   0.3 /  0.1    9.9  done
   43.9   6.9 /  6.9   15.9  episode end

SAROPS, task, lanes no overlap: tasks (plan min, transit est. / actual min, length min, end)
    2.0   9.5 /  9.2   20.7  done
   22.7   1.0 /  0.8   11.3  done
   34.0   6.8 /  6.8   16.5  done
   50.5   0.1 /  0.1    9.4  episode end

scenario 301, sweep width 203 m, search 10 min, max transit 10 min

| planner | POS | held-out POS | POS at 15 / 30 min | time not on lanes | first lane (min after handover) | plans | jumps > 3 km | s |
|---|---|---|---|---|---|---|---|---|
| accordion, task, lanes no overlap | 0.415 | 0.405 | 0.102 / 0.212 | 28% | 8.4 | 4 | 1 | 1 |
| SAROPS, task, lanes no overlap | 0.465 | 0.460 | 0.154 / 0.331 | 29% | 9.2 | 4 | 1 | 28 |
rl_search/data/task_mode/task_mode_s301_x10_fixed.png

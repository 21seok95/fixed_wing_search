
accordion, task, lanes k=1-3: tasks (plan min, transit est. / actual min, length min, end)
    4.5  10.2 / 10.0   19.7  done  2.8x0.9 km k=1
   24.2   2.4 /  2.2   13.6  done  1.4x1.7 km k=1
   37.8   4.7 /  4.7   14.2  done  1.2x0.9 km k=2
   52.0   4.9 /  4.7    7.9  episode end  1.2x0.9 km k=2
  approach phases (min): 2.0-4.5

SAROPS, task, lanes k=1-3: tasks (plan min, transit est. / actual min, length min, end)
    4.5  10.2 / 10.0   19.7  done  2.8x0.9 km k=1
   24.2   4.4 /  4.2   15.5  done  1.4x1.7 km k=1
   39.7   2.9 /  2.8   12.2  done  1.2x0.9 km k=2
   51.9   6.3 /  6.1    8.0  episode end  1.2x0.9 km k=1
  approach phases (min): 2.0-4.5

scenario 302, sweep width 203 m, search 10 min, max transit 10 min

| planner | POS | held-out POS | POS at 15 / 30 min | time not on lanes | first lane (min after handover) | plans | jumps > 3 km | s |
|---|---|---|---|---|---|---|---|---|
| accordion, task, lanes k=1-3 | 0.295 | 0.285 | 0.010 / 0.153 | 41% | 12.4 | 4 | 3 | 1 |
| SAROPS, task, lanes k=1-3 | 0.323 | 0.313 | 0.010 / 0.166 | 44% | 12.4 | 4 | 3 | 27 |
rl_search/data/task_mode/task_mode_s302_x10_cover.png

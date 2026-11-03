
accordion, task: tasks (plan min, transit est. min, length min, end)
    2.0   8.4   16.2  done
   18.2   1.1   10.3  done
   28.5   6.6   16.0  done
   44.5   6.8   15.1  done
   59.6   0.5    0.3  episode end

SAROPS, task: tasks (plan min, transit est. min, length min, end)
    2.0   8.6   18.2  done
   20.2   0.9   10.3  done
   30.6   5.4    8.8  done
   39.4   5.3   13.3  done
   52.8   6.1    7.2  episode end

scenario 301, sweep width 203 m, search 10 min, max transit 10 min

| planner | POS | held-out POS | POS at 15 / 30 min | time not on lanes | first lane (min after handover) | plans | jumps > 3 km | s |
|---|---|---|---|---|---|---|---|---|
| accordion, 10-min cycle | 0.483 | 0.476 | 0.181 / 0.323 | 40% | 9.8 | 6 | 2 | 1 |
| accordion, task | 0.492 | 0.482 | 0.102 / 0.328 | 41% | 8.3 | 5 | 2 | 1 |
| SAROPS, 10-min cycle | 0.487 | 0.483 | 0.190 / 0.340 | 60% | 10.0 | 6 | 4 | 35 |
| SAROPS, task | 0.507 | 0.498 | 0.220 / 0.373 | 49% | 8.9 | 5 | 3 | 36 |
rl_search/data/task_mode/task_mode_s301_x10.png

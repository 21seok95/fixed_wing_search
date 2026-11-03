
accordion, task: tasks (plan min, transit est. min, length min, end)
    4.0  10.0   19.8  done
   23.8   5.5   13.8  done
   37.5   0.7    8.5  done
   46.0   3.9   13.1  done
   59.1   5.7    0.8  episode end
  approach phases (min): 2.0-4.0

SAROPS, task: tasks (plan min, transit est. min, length min, end)
    4.0  10.0   22.0  cap
   26.0   3.6   11.6  done
   37.6   1.7    9.7  done
   47.2   5.2    9.3  done
   56.6   0.6    3.3  episode end
  approach phases (min): 2.0-4.0

scenario 302, sweep width 203 m, search 10 min, max transit 10 min

| planner | POS | held-out POS | POS at 15 / 30 min | time not on lanes | first lane (min after handover) | plans | jumps > 3 km | s |
|---|---|---|---|---|---|---|---|---|
| accordion, 10-min cycle | 0.312 | 0.303 | 0.052 / 0.123 | 48% | 13.0 | 6 | 3 | 1 |
| accordion, task | 0.242 | 0.235 | 0.008 / 0.072 | 41% | 12.2 | 5 | 3 | 1 |
| SAROPS, 10-min cycle | 0.366 | 0.361 | 0.067 / 0.169 | 56% | 13.4 | 6 | 3 | 29 |
| SAROPS, task | 0.325 | 0.319 | 0.057 / 0.104 | 46% | 14.8 | 5 | 3 | 30 |
rl_search/data/task_mode/task_mode_s302_x10.png

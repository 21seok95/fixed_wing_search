
accordion, task: tasks (plan min, transit est. / actual min, length min, end)
    4.0  10.0 / 10.2   19.8  done
   23.8   5.5 /  5.9   13.8  done
   37.5   0.7 /  0.6    8.5  done
   46.0   3.9 /  4.2   13.1  done
   59.1   5.7 /   -     0.8  episode end
  approach phases (min): 2.0-4.0

SAROPS, task: tasks (plan min, transit est. / actual min, length min, end)
    4.0  10.0 / 12.8   22.0  cap
   26.0   3.6 /  3.8   11.6  done
   37.6   1.7 /  1.7    9.7  done
   47.2   5.2 /  5.2    9.3  done
   56.6   0.6 /  1.6    3.3  episode end
  approach phases (min): 2.0-4.0

accordion, task, peaks at arrival: tasks (plan min, transit est. / actual min, length min, end)
    4.0  10.0 / 10.2   19.8  done
   23.8   5.6 /  6.0   14.8  done
   38.5   2.1 /  2.2   10.1  done
   48.6   3.0 /  2.9   10.7  done
   59.2   1.5 /   -     0.7  episode end
  approach phases (min): 2.0-4.0

SAROPS, task, peaks at arrival: tasks (plan min, transit est. / actual min, length min, end)
    4.0  10.0 / 12.8   22.0  cap
   26.0   4.3 /  6.1   15.0  done
   41.0   3.8 /  4.5   13.4  done
   54.4   4.0 /   -     5.5  episode end
  approach phases (min): 2.0-4.0

scenario 302, sweep width 203 m, search 10 min, max transit 10 min

| planner | POS | held-out POS | POS at 15 / 30 min | time not on lanes | first lane (min after handover) | plans | jumps > 3 km | s |
|---|---|---|---|---|---|---|---|---|
| accordion, task | 0.242 | 0.235 | 0.008 / 0.072 | 41% | 12.2 | 5 | 3 | 1 |
| SAROPS, task | 0.325 | 0.319 | 0.057 / 0.104 | 46% | 14.8 | 5 | 3 | 50 |
| accordion, task, peaks at arrival | 0.271 | 0.271 | 0.008 / 0.060 | 41% | 12.2 | 5 | 3 | 1 |
| SAROPS, task, peaks at arrival | 0.303 | 0.290 | 0.057 / 0.106 | 53% | 14.8 | 4 | 3 | 34 |
rl_search/data/task_mode/task_mode_s302_x10_arrival.png

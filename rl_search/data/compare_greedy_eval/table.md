New sensor + lrc. POS held-out = same actions on particles the planner never saw.

## all: 6 scenarios

| method | held-out POS mean ± sd | median | min | POS on planner's particles | s/episode |
|---|---|---|---|---|---|
| SAROPS | 0.518 ± 0.151 | 0.552 | 0.286 | 0.513 | 46 |
| greedy90 | 0.514 ± 0.201 | 0.542 | 0.230 | 0.519 | 71 |
| accordion | 0.493 ± 0.152 | 0.510 | 0.251 | 0.490 | 1 |

SAROPS − greedy90 (paired, mean ± se | wins): +0.0038 ± 0.0240 | 4/6
SAROPS − accordion (paired, mean ± se | wins): +0.0245 ± 0.0101 | 5/6

## new (308+): 6 scenarios

| method | held-out POS mean ± sd | median | min | POS on planner's particles | s/episode |
|---|---|---|---|---|---|
| SAROPS | 0.518 ± 0.151 | 0.552 | 0.286 | 0.513 | 46 |
| greedy90 | 0.514 ± 0.201 | 0.542 | 0.230 | 0.519 | 71 |
| accordion | 0.493 ± 0.152 | 0.510 | 0.251 | 0.490 | 1 |

SAROPS − greedy90 (paired, mean ± se | wins): +0.0038 ± 0.0240 | 4/6
SAROPS − accordion (paired, mean ± se | wins): +0.0245 ± 0.0101 | 5/6

## Per scenario, held-out POS

| seed | SAROPS | greedy90 | accordion |
|---|---|---|---|
| 1000000 | 0.534 | 0.532 | 0.520 |
| 1000001 | 0.570 | 0.553 | 0.500 |
| 1000002 | 0.286 | 0.230 | 0.251 |
| 1000003 | 0.384 | 0.312 | 0.384 |
| 1000004 | 0.759 | 0.845 | 0.743 |
| 1000005 | 0.575 | 0.614 | 0.562 |

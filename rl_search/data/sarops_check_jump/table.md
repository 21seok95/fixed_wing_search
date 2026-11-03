Scenarios: [300, 301, 302, 303, 304, 305, 306, 307]. New sensor + lrc, accordion tavg_pca/200 with 3 peaks, 10-min cycle.
POS on the planner's particles / on held-out particles (same actions).

| method | POS mean ± sd | min | held-out POS mean ± sd | min | held-out vs accordion | wins | s/episode | evals/cycle | moves used |
|---|---|---|---|---|---|---|---|---|---|
| accordion | 0.369 ± 0.130 | 0.209 | 0.369 ± 0.132 | 0.202 | +0.000 (+0.0%) | 0/8 | 1 | 0 | - |
| rotate+corner+spacing | 0.393 ± 0.123 | 0.227 | 0.388 ± 0.127 | 0.215 | +0.019 (+5.3%) | 8/8 | 20 | 36 | {'rotate': 49, 'corner': 30, 'spacing': 67} |
| big jump | 0.409 ± 0.122 | 0.259 | 0.405 ± 0.127 | 0.242 | +0.037 (+9.9%) | 8/8 | 49 | 92 | {'corner': 32, 'spacing': 75, 'rotate': 43} |

Per scenario, held-out POS:

| seed | accordion | rotate+corner+spacing | big jump |
|---|---|---|---|
| 300 | 0.202 | 0.215 | 0.242 |
| 301 | 0.476 | 0.485 | 0.483 |
| 302 | 0.303 | 0.345 | 0.361 |
| 303 | 0.239 | 0.266 | 0.267 |
| 304 | 0.354 | 0.363 | 0.368 |
| 305 | 0.336 | 0.360 | 0.359 |
| 306 | 0.395 | 0.416 | 0.507 |
| 307 | 0.644 | 0.655 | 0.655 |

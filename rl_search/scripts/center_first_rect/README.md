# First-rectangle overlap: diagnosis and fixes (2026-11-06)

Exploratory scripts, run from the repository root with `PYTHONPATH=.`. The data and figures are in
`rl_search/data/center_open/first_rect/` (diagnosis) and `rl_search/data/center_open/first_fix/` (fixes). Background:
the 2026-11-06 section of `docs/sarops_approach_todo.md`.

Diagnosis. These scripts use the saved flights only, except `diag_first.py`, which reruns the planner with logging and
checks that each flight matches the saved one.

```bash
D=rl_search/data/center_open/first_rect
python rl_search/scripts/center_first_rect/decomp.py $D/decomp.json                  # where the double-swept area comes from
python rl_search/scripts/center_first_rect/extra.py $D/extra.json                    # time by phase, close time by round, rectangle sizes
python rl_search/scripts/center_first_rect/fig.py $D/decomp.json $D/extra.json $D/overlap_causes.png
python rl_search/scripts/center_first_rect/diag_first.py --out $D/diag.jsonl --workers 9   # instrumented reruns (~20 min)
python rl_search/scripts/center_first_rect/flow.py $D/diag.jsonl $D/flow.json        # particles kept in each rectangle
python rl_search/scripts/center_first_rect/fig2.py $D/diag.jsonl $D/first_rect_stats.png $D/first_rect_map.png $D/flow.json
python rl_search/scripts/center_first_rect/explain.py $D/diag.jsonl $D/explain.json  # density, missed particles, peak time
python rl_search/scripts/center_first_rect/fig3.py $D/explain.json $D/diag.jsonl $D/explain_missed.png $D/explain_peak.png
```

Fixes, with 5 UAVs, A + POS, x 70, w 1, both with no constraint and with both rules. The fixes are options, so the
defaults reproduce the earlier runs.

```bash
python rl_search/scripts/center_first_rect/run_fix.py                                # 60 flights (~30 min, 9 workers)
python -m rl_search.scripts.compare_center_open --sweep --peak-intercept --adjacent-lanes --sep --no-overlap --n 5 --xs 70 --ws 1   # table.md, no new runs
python rl_search/scripts/center_first_rect/fix_summary.py $D/fix_rows.json
python rl_search/scripts/center_first_rect/fix_fig.py $D/fix_rows.json rl_search/data/center_open/first_fix/fix_metrics.png rl_search/data/center_open/first_fix/fix_map_504.png
```

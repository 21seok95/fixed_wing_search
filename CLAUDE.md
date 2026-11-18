# fixed_wing_search — branch local-pc (the local PC)

This branch holds only the code the local PC needs (training, evaluation, analysis, the web viewer). It is made from `feat/rl-env` on the
Mac with `rl_search/scripts/sync_local_pc.py` and has no shared history with it: never merge it into `feat/rl-env`, and do not commit
results here. Bring results back as the zip from `python -m rl_search.scripts.rl_eval.export_run --run <run>`. It replaced
`local-pc-r13` on 2026-11-18 (that branch is left as it was).

Kept small on purpose (user 2026-11-18): no model weights, training logs, TensorBoard files, web runs, figures or reports; no DEM or land
cover raw data (fixed releases, downloaded again by make_real_terrain). The OSM downloads are here gzipped (`raw/osm_*.json.gz`), so the
maps are built from the Mac's download. After building a map, run `python -m rl_search.scripts.terrain_fingerprint check --region ...`:
it must say "same as the Mac" before any training on that map.

Read `local_pc.md` section 0 first (the current work, commands in order with checks, what to report). Install with `docs/run_on_pc.md`
section 1 (skip its clone and checkout lines). Every RL run is logged in `docs/rl_experiments.md`; the MDP design is in
`docs/rl_mdp_design.md`; the training-scenario redesign after R13 (maps, starts, particle rules, observation) is in
`docs/scenario_redesign.md`.

Working rules agreed with the user (`local_pc.md` section 7): reply in Korean; show every result as figures as well as tables; say in
plain words what labels such as R10, A′ or nval mean; go one step at a time and check with the user; keep settings marked "바꾸지 말 것";
commit or push only when the user asks.

`rl_search/data/` and `*.png` are git-ignored. Run scripts from the repository root with bash. The web viewer
`python -m rl_search.scripts.real_map.vehicle_web` (http://127.0.0.1:8000) has every feature added on the Mac; start it after training
(an evaluation with the web viewer on once ran out of memory).

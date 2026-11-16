# fixed_wing_search — branch local-pc-r13 (R13 on the local PC)

This branch holds only the files R13 needs on the local PC. It was cut from `feat/rl-env` on 2026-11-16 as one commit with no shared
history. Do not merge it into `feat/rl-env`; bring results back as the zip from `python -m rl_search.scripts.rl_eval.export_run --run r13_2map`.

Read `local_pc.md` section 0 first: background since R8, what R13 changes and why, the commands in order with checks, what to analyse
and what to report. Install with `docs/run_on_pc.md` section 1 (skip its clone and checkout lines). Every RL run is logged in
`docs/rl_experiments.md`; the MDP design agreed with the user is in `docs/rl_mdp_design.md`; the R10-R12 set-up and results are in
`docs/r10_ubuntu.md` sections 8 and 9.

Working rules agreed with the user (`local_pc.md` section 7): reply in Korean; show every result as figures as well as tables; say in
plain words what labels such as R10, A′ or nval mean; go one step at a time and check with the user; keep the R13 settings marked
"바꾸지 말 것"; commit or push only when the user asks.

`rl_search/data/` and `*.png` are git-ignored, so results are added with `git add -f`. Build the two terrains once per machine with
`python -m rl_search.scripts.make_real_terrain --region amboy --cell 25` and the same with `--region mojave` (the raw data is in this
branch, nothing is downloaded). Run scripts from the repository root with bash.

The web viewer `python -m rl_search.scripts.real_map.vehicle_web` (http://127.0.0.1:8000) replays the saved validation flights and flies
any checkpoint on the page's Amboy scenario. Start it after training: an evaluation with the web viewer on once ran out of memory.

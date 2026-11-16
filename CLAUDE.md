# fixed_wing_search

Start with the newest dated section at the top of `docs/sarops_approach_todo.md`. It holds the goal, the decisions agreed
with the user, the latest results, the next steps and the open questions. Work through experiments one step at a time and
check with the user between steps. Show every result as figures as well as tables, even if that takes longer.
The real-region map work (Chungju, open data, road graph) has its own handoff in `docs/real_map_todo.md`; read its newest
section too. The RL redesign agreed with the user item by item (MDP, model, training settings, validation set) is in
`docs/rl_mdp_design.md`; its last section says how to continue on the Ubuntu PC. Every RL experiment (what changed, why, result, next
decision) is logged in `docs/rl_experiments.md`, with names for the A definitions and reward versions; update it whenever a run starts or ends.

**On the Ubuntu PC**, the next run (R10: targets that never stop, new map channels, no A) is set up in `docs/r10_ubuntu.md`: read it
first. It holds the reasons, the decisions, the commands in order with checks, and what to report. Section 8 has the R10, R11 and R12 results
(2026-11-15) and how to continue the analysis on the Mac (what is in git, what is not, the commands); section 9 sets up R13
(a fifth 102.4 km map and the look reward A' at 0.1, decided 2026-11-16, with R10's PPO settings); R13 is run on the local PC from the slim branch `local-pc-r13` (only the files R13 needs, one commit with no shared history; never
merge it into feat/rl-env), see `local_pc.md` section 0.

**On the local PC** (a machine with no connection to the main work environment), read `local_pc.md` first; its section 0 is the current
work (R13: background since R8, what changes and why, commands in order with checks, what to analyse and report). The rest
holds the earlier R8 and R7 hand-offs: the working rules agreed with the user, the order of work with checks, and what to
leave behind. The commands are in `docs/run_on_pc.md`; results are packed into one zip with `scripts/rl_eval/export_run.py`.

`rl_search/data/` and `*.png` are git-ignored, so results are added with `git add -f`. The terrain file is not in git:
run `python -m rl_search.scripts.make_terrain` once per machine. Real-region terrains are rebuilt with
`python -m rl_search.scripts.make_real_terrain --region <key> --cell 25` (Amboy's raw downloads are in git, so it rebuilds the
same map offline; Chungju and Mojave download theirs). Run scripts from the repository root.

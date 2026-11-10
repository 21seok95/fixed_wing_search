# fixed_wing_search

Start with the newest dated section at the top of `docs/sarops_approach_todo.md`. It holds the goal, the decisions agreed
with the user, the latest results, the next steps and the open questions. Work through experiments one step at a time and
check with the user between steps. Show every result as figures as well as tables, even if that takes longer.
The real-region map work (Chungju, open data, road graph) has its own handoff in `docs/real_map_todo.md`; read its newest
section too. The RL redesign agreed with the user item by item (MDP, model, training settings, validation set) is in
`docs/rl_mdp_design.md`; its last section says how to continue on the Ubuntu PC.

`rl_search/data/` and `*.png` are git-ignored, so results are added with `git add -f`. The terrain file is not in git:
run `python -m rl_search.scripts.make_terrain` once per machine. Real-region terrains are rebuilt with
`python -m rl_search.scripts.make_real_terrain --region <key> --cell 25` (Amboy's raw downloads are in git, so it rebuilds the
same map offline; Chungju and Mojave download theirs). Run scripts from the repository root.

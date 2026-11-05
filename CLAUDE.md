# fixed_wing_search

Start with the newest dated section at the top of `docs/sarops_approach_todo.md`. It holds the goal, the decisions agreed
with the user, the latest results, the next steps and the open questions. Work through experiments one step at a time and
check with the user between steps.

`rl_search/data/` and `*.png` are git-ignored, so results are added with `git add -f`. The terrain file is not in git:
run `python -m rl_search.scripts.make_terrain` once per machine. Run scripts from the repository root.

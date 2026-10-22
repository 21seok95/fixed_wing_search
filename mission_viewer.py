"""Interactive replay of UAV search missions over the time-evolving probability density.

    python uav_search.py results_battlefield_paper --plan greedy      # writes search_greedy/mission.npz
    python uav_search.py results_battlefield_paper --plan tree        # writes search_tree/mission.npz
    python mission_viewer.py results_battlefield_paper                # pick any mission inside the window
    python mission_viewer.py results_battlefield_paper --select tree --compare greedy straight
    python mission_viewer.py results_battlefield_paper/search_tree --time 90 --snapshot m.png

Every search_* folder of the run is listed under "View mission"; "Compare POS with"
overlays other missions' POS curves.
Density modes: Prior (no search), Remaining (weights after detections, mass = 1 - POS),
Conditional (remaining renormalised to 1, i.e. where the target is given it is not yet found).
The KDE bandwidth is fixed to the one the planner used, so the picture matches the logged POS.
Keys: left/right = one mission step, up/down = previous/next mission, space = play/pause.
"""
import sys
import matplotlib

if "--snapshot" in sys.argv:
    matplotlib.use("Agg")

from sim_viewer import main  # noqa: E402

if __name__ == "__main__":
    main(mission_mode=True)

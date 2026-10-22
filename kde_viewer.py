"""Interactive viewer for the terrain-masked KDE over time.

    python kde_viewer.py results_battlefield_paper
    python kde_viewer.py results_v2 --time 60 --bandwidth 300
    python kde_viewer.py results_battlefield_paper --time 120 --snapshot kde.png

Controls: time slider, h_s (bandwidth) slider, layer check boxes, Play button.
Keys: left/right = time step, up/down = bandwidth +-25 m, space = play/pause.
"""
import sys
import matplotlib

if "--snapshot" in sys.argv:
    matplotlib.use("Agg")

from sim_viewer import main  # noqa: E402

if __name__ == "__main__":
    main(mission_mode=False)

# rl_search

Lawnmower vs RL heading search for a fixed-wing UAV on a fixed 60 km x 60 km terrain.
The root scripts (`uav_search.py`, `battlefield_infiltrator_demo.py`, viewers) are kept
unchanged as the reference; the parts that need changes are forked here.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r rl_search/requirements.txt
```

## Step 0: terrain

Generate the terrain once (about 20 s; output is not tracked by git):

```bash
.venv/bin/python -m rl_search.scripts.make_terrain
```

The terrain is procedural: ridged mountain ranges on uplands, rolling hills, rivers and
streams from the drainage network of the elevation (deep basins become lakes, shallow ones
sediment-filled plains), fords along the rivers, and forest on high and steep ground.
Mountains above the no-go elevation (top 8% of the map), rivers, lakes and slopes of 32
degrees or more are impassable for the target.

This writes `rl_search/data/terrain_seed7_60km.npz` (base layers only; slopes, cover and
clearance are derived on load by `rl_search.terrain.load_terrain`) and an overview figure
with the 20 / 40 / 60 km curriculum areas.

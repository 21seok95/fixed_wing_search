"""Replay the area score runs in the sim_viewer window: time slider, play button, density modes, POS panel, keys.

Saved flights are converted to the files sim_viewer reads, with no planner rerun, into
rl_search/data/center_open/view/s<seed>_n<n>/: simulation_data.npz (held-out particles and terrain) and one
search_*/mission.npz per flight. The first run for a scenario takes under a minute. On top of sim_viewer this draws the
moving x% region of the unweighted particles and the current SAROPS rectangles (the "Planned path" check box).

    python -m rl_search.scripts.view_center_open --seed 502 --n 5 --x 70 --w 1
    python -m rl_search.scripts.view_center_open --seed 502 --n 5 --x 85 --w 1   # adds this flight to the same list

Keys: left/right = one step, up/down = previous/next flight, space = play/pause. Weights come from the 10 s paths and
include the launch transit, so the POS shown can differ a little from the table.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import warnings

import matplotlib
if "--snapshot" in sys.argv or "--check" in sys.argv:
    matplotlib.use("Agg")
from matplotlib.lines import Line2D
import numpy as np

import sim_viewer
from sim_viewer import SimViewer, crop_run, find_missions, load_run

SWEEP = Path("rl_search/data/center_open/sweep")
OUT = Path("rl_search/data/center_open/view")
XS = (50.0, 70.0, 85.0)
REGION_COLOR = "#00e676"
sim_viewer.UAV_COLORS = ("#e040fb", "#00e5ff", "#ffea00", "#ff3d00", "#ffffff")   # five UAVs, one colour each


def mission_arrays(env, hold, r) -> dict:
    """Poses and team weights on a common 10 s grid. UAVs wait at the launch point before launch."""
    from rl_search.mission import ParticleBelief
    xy, tt = [np.array(p, float) for p in r["xy"]], [np.array(t, float) for t in r["tt"]]
    if any(np.any(np.abs(t/10.0 - np.round(t/10.0)) > 1e-6) for t in tt):
        raise ValueError("expected paths on a 10 s grid")
    grid = np.arange(0.0, max(t[-1] for t in tt) + 1.0, 10.0)
    n_uav = len(xy)
    poses = np.zeros((len(grid), n_uav, 3))
    for u in range(n_uav):
        px, py = np.interp(grid, tt[u], xy[u][:, 0]), np.interp(grid, tt[u], xy[u][:, 1])
        dx, dy = np.diff(px, append=px[-1]), np.diff(py, append=py[-1])
        moving = np.hypot(dx, dy) > 1e-6
        yaw = np.arctan2(dy, dx)
        if moving.any() and not moving.all():
            on, off = np.flatnonzero(moving), np.flatnonzero(~moving)
            yaw[off] = yaw[on[np.searchsorted(on, off).clip(max=len(on) - 1)]]
        poses[:, u] = np.stack([px, py, yaw], 1)
    env.belief = ParticleBelief(hold.track, hold.save_dt_s, hold.exit_step, True)
    n = env.belief.n
    w = [np.ones(n) for _ in range(n_uav)]
    state = [(np.zeros(n), np.full(n, -np.inf)) for _ in range(n_uav)]
    done = [0]*n_uav
    weights = np.ones((len(grid), n), np.float32)
    for k, t in enumerate(grid):
        for u in range(n_uav):
            while done[u] + 1 < len(tt[u]) and tt[u][done[u] + 1] <= t + 1e-6:
                i = done[u]
                env._detect_lrc(xy[u][i:i + 2], tt[u][i:i + 2], w[u], state[u])
                done[u] += 1
        weights[k] = np.prod(w, axis=0)
    rects = [[u, *q] for u, rs in enumerate(r["rects"]) for q in rs]
    return dict(t_s=grid, poses=poses.astype(np.float32), weights=weights, pos_cum=(1 - weights.mean(1)).astype(np.float64),
                plan=np.full((len(grid), n_uav, 1, 3), np.nan, np.float32), rects=np.array(rects, float).reshape(-1, 8))


def export(seed, n, x, w, source=SWEEP, out=OUT) -> Path:
    from rl_search.area_score import region_masks
    from rl_search.scripts.center_open import open_scenario
    from rl_search.scripts.check_sarops import holdout_scenario
    from rl_search.scripts.compare_center_open import make_envs
    run_dir = out/f"s{seed}_n{n}"
    wanted = {"pos_only": lambda r: r["config"] == "pos",
              f"score_x{x:g}_w{w:g}": lambda r: r["config"] == "score" and r["x"] == x and r["w"] == w}
    missing = {k: f for k, f in wanted.items() if not (run_dir/f"search_{k}"/"mission.npz").exists()}
    if not missing and (run_dir/"simulation_data.npz").exists():
        return run_dir
    print(f"Converting scenario {seed}, {n} UAVs for the viewer, up to a few minutes when the machine is busy...", flush=True)
    rows = [r for r in map(json.loads, (source/"results.jsonl").read_text().splitlines()) if r["seed"] == seed and r["n"] == n]
    env = make_envs(1, seed)[0]
    scen = open_scenario(env.sampler, seed)
    hold = holdout_scenario(env, scen, seed + 10**6)
    run_dir.mkdir(parents=True, exist_ok=True)
    if not (run_dir/"simulation_data.npz").exists():
        terr = env.terr
        regions = {}
        for xx in XS:
            bits = np.zeros(terr.forbidden.shape, np.uint16)
            for j, m in enumerate(region_masks(terr, env.h, hold.track, hold.save_dt_s, xx)):
                bits |= m.astype(np.uint16) << j
            regions[f"region_x{int(xx)}"] = bits
        np.savez_compressed(run_dir/"simulation_data.npz", axis=terr.axis, track=np.asarray(hold.track, np.float32), dt_s=hold.save_dt_s,
                            lkp=scen.lkp, heading_deg=scen.heading_deg, terrain=terr.z, mountain=terr.mountain.astype(bool), water=terr.water.astype(bool),
                            forest=terr.forest.astype(bool), fords=terr.fords.astype(bool), forbidden=terr.forbidden.astype(bool), **regions)
    for name, f in missing.items():
        r = next((q for q in rows if f(q)), None)
        if r is None:
            raise SystemExit(f"No run for seed {seed}, {n} UAVs, {name} in {source}")
        arrays = mission_arrays(env, hold, r)
        meta = dict(planner="SAROPS, POS only" if r["config"] == "pos" else f"SAROPS, A + {w:g} POS", region_x=x if r["config"] == "score" else None,
                    uav_speed_mps=float(env.v), dt_s=10.0, max_turn_deg=30.0, n_headings="-", flight_min=60, start_x=float(r["start"][0]), start_y=float(r["start"][1]),
                    fov_along_m=float(env.fp_along), fov_cross_m=float(env.fp_cross), bandwidth_m=float(env.h), pod=float(env.search.pod),
                    table_pos=r["pos"], c={k: v["c"] for k, v in r["scores"].items()}, close_s=r["below"],
                    planner_opts={"POS in table": round(r["pos"], 3), "pair < 300 m (s)": round(r["below"])})
        d = run_dir/f"search_{name}"
        d.mkdir(exist_ok=True)
        np.savez_compressed(d/"mission.npz", meta=json.dumps(meta), **arrays)
        print(f"{name}: POS from the 10 s path {arrays['pos_cum'][-1]:.3f}, in the table {r['pos']:.3f}", flush=True)
    return run_dir


class CenterViewer(SimViewer):
    """sim_viewer with the moving x% region and the current rectangles."""

    def __init__(self, *args, x=70.0, **kw):
        self.x, self._region = x, None
        super().__init__(*args, **kw)

    def _init_mission_artists(self, current):
        return super()._init_mission_artists(current) + [Line2D([], [], color=REGION_COLOR, lw=1.8, label="Target x% region (no search weights)")]

    def update(self):
        super().update()
        if not self.ready or self.mission is None:
            return
        t_s = float(self.s_time.val)*60
        m = self.mission
        x = m["meta"].get("region_x") or self.x
        if self._region is not None:
            self._region.remove()
            self._region = None
        j = int(np.clip(np.ceil(t_s/300.0 - 1e-9), 1, 12)) - 1
        mask = (self.run[f"region_x{int(x)}"] >> j) & 1
        extra = f"\n{x:g}% region {mask.sum()*(self.run['cell']/1000)**2:.1f} km²"
        if mask.any():
            self._region = self.ax.contour(self.xa, self.ya, mask.astype(float), levels=[.5], colors=REGION_COLOR, linewidths=1.8, zorder=8)
        c = m["meta"]["c"].get(str(int(x)))
        if c is not None and t_s >= 300.0:
            extra += f", swept {c[j]:.0%} (table)"
        self.info.set_text(self.info.get_text() + extra)
        polys = []
        if self.chk.get_status()[5] and len(m["rects"]) and t_s >= self.t_launch:
            for u in range(self.n_uav):
                own = m["rects"][(m["rects"][:, 0] == u) & (m["rects"][:, 1] <= t_s)]
                if len(own):
                    _, _, cx, cy, ell, wid, th, _ = own[-1]
                    c_, s_ = np.cos(th), np.sin(th)
                    corners = np.array([[a*ell/2, b*wid/2] for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
                    polys.append((np.array([cx, cy]) + corners @ np.array([[c_, s_], [-s_, c_]]))/1000)
        self.plan_coll.set_verts(polys)
        self.fig.canvas.draw_idle()


def main():
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=502)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--x", type=float, default=70.0)
    ap.add_argument("--w", type=float, default=1.0)
    ap.add_argument("--source", type=Path, default=SWEEP)
    ap.add_argument("--time", type=float, default=None, help="initial time in minutes")
    ap.add_argument("--mode", choices=sim_viewer.DENSITY_MODES, default=None)
    ap.add_argument("--margin", type=float, default=2000.0)
    ap.add_argument("--snapshot", type=Path, default=None, help="save a PNG and exit")
    ap.add_argument("--check", action="store_true", help="drive every control without a window")
    args = ap.parse_args()
    run_dir = export(args.seed, args.n, args.x, args.w, args.source)
    run = load_run(run_dir)
    missions = find_missions(run_dir)
    names = [p.parent.name.removeprefix("search_") for p in missions]
    pts = np.concatenate([run["track"][:int(3600/run["dt_s"]) + 1].reshape(-1, 2)] + [np.load(p)["poses"][0, :, :2] for p in missions])
    crop_run(run, pts, args.margin)
    v = CenterViewer(run, missions, names.index(f"score_x{args.x:g}_w{args.w:g}"), args.time, x=args.x)
    if args.mode:
        v.radio.set_active(sim_viewer.DENSITY_MODES.index(args.mode))
    for i, name in enumerate(names):
        if i != v.cur:
            v.chk_cmp.set_active(i)
    if args.check:
        for t in (0.0, 3.0, 5.0, 15.0, 30.0, 45.0, 60.0, 65.0):
            v.s_time.set_val(t)
            v.fig.canvas.draw()
        for mode in sim_viewer.DENSITY_MODES:
            v.radio.set_active(sim_viewer.DENSITY_MODES.index(mode))
            v.fig.canvas.draw()
        for i in range(len(names)):
            v.radio_mis.set_active(i)
            v.s_time.set_val(30.0)
            v.fig.canvas.draw()
            print(f"{names[i]}: UAVs {v.n_uav}, steps {len(v.mission['t_s'])}, rectangles {len(v.mission['rects'])}, final POS {v.mission['pos_cum'][-1]:.3f}")
        for lbl in ("Particles", "Searched swath", "Planned path", "Log color"):
            v.chk.set_active([t.get_text() for t in v.chk.labels].index(lbl))
            v.fig.canvas.draw()
        v.toggle_play()
        v._tick()
        v.toggle_play()
        print("every control ran without an error")
        return
    if args.snapshot is not None:
        v.fig.savefig(args.snapshot, dpi=130)
        print("saved", args.snapshot)
        return
    print("Window open. If you do not see it, look for the Python icon in the Dock. Close the window to return here.", flush=True)
    sim_viewer.plt.show()


if __name__ == "__main__":
    main()

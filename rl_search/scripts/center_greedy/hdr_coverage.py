"""Exploratory analysis, no planning rerun: share of the 85% region of the unweighted particle distribution at t = 3600 s
that the UAVs' sensors swept at any time in the flight. Repeat passes count once, POD is not used.

Region: rl_env._zone with hdr_level = 0.85 and no margin (smallest area holding 85% of the smoothed density, no-go cells
excluded), on the 25 m grid, from the held-out particles. Swept: a 25 m cell whose centre is within 143 m of the path.
"""
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, REPO)
os.chdir(REPO)

import json
import warnings
from multiprocessing import Pool
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(REPO)/"rl_search/data/center_greedy"
SEEDS = list(range(400, 410))
METHODS = ("sarops", "greedy")
NS = (1, 3, 5)
T_REGION, T_H, TARGET = 3600.0, 120.0, .85
MINUTES = (15, 30, 45, 60)


def load_runs():
    new = {(r["seed"], r["method"], r["n"], r["sep"]): r for r in map(json.loads, open(HERE/"center_greedy.jsonl"))}
    old = {(r["seed"], r["mode"]): r for r in json.load(open("rl_search/data/compare_center/results.json"))}

    def get(seed, method, n):
        if method == "sarops" and n in (1, 5):
            r = old[(seed, "1" if n == 1 else "5 no sep")]
            return dict(r, hold=r["hold_pos"])
        return new[(seed, method, n, False)]
    return get


def sweep(first, lo, c, xy, tt, R):
    """Earliest time each cell centre comes within R of the polyline (updates ``first`` in place)."""
    n = first.shape[0]
    m = int(np.ceil(R/c)) + 1
    for k in range(len(tt) - 1):
        a, b = xy[k], xy[k + 1]
        ix0, ix1 = int(np.floor((min(a[0], b[0]) - lo)/c)) - m, int(np.floor((max(a[0], b[0]) - lo)/c)) + m + 1
        iy0, iy1 = int(np.floor((min(a[1], b[1]) - lo)/c)) - m, int(np.floor((max(a[1], b[1]) - lo)/c)) + m + 1
        ix0, iy0, ix1, iy1 = max(ix0, 0), max(iy0, 0), min(ix1, n), min(iy1, n)
        if ix0 >= ix1 or iy0 >= iy1:
            continue
        X, Y = np.meshgrid(lo + (np.arange(ix0, ix1) + .5)*c, lo + (np.arange(iy0, iy1) + .5)*c)
        ab = b - a
        s = np.clip(((X - a[0])*ab[0] + (Y - a[1])*ab[1])/max(ab @ ab, 1e-9), 0, 1)
        hit = (X - a[0] - s*ab[0])**2 + (Y - a[1] - s*ab[1])**2 <= R*R
        view = first[iy0:iy1, ix0:ix1]
        view[hit] = np.minimum(view[hit], (tt[k] + s*(tt[k + 1] - tt[k]))[hit])


def check_sweep():
    """Known areas: a straight 9.9 km path off the grid axes (a path along an axis shows the worst 25 m rounding of the
    width: 300 m instead of 286 m), and a full circle at the turn limit sampled at 1 s and at 10 s."""
    c, lo, n, R = 25.0, -8000.0, 640, 143.0
    t = np.arange(0, 667, 10.0)
    u = np.array([np.cos(np.deg2rad(17.0)), np.sin(np.deg2rad(17.0))])
    first = np.full((n, n), np.inf, np.float32)
    sweep(first, lo, c, (t*15 - 4950)[:, None]*u, t, R)
    line = np.isfinite(first).sum()*c*c/1e6
    exact = 9.9*2*R/1e3 + np.pi*(R/1e3)**2
    out = {}
    for dt in (1.0, 10.0):
        t = np.arange(0, 60 + dt, dt)
        th = t*np.deg2rad(6)
        first = np.full((n, n), np.inf, np.float32)
        sweep(first, lo, c, np.stack([143*np.cos(th), 143*np.sin(th)], 1), t, R)
        out[dt] = np.isfinite(first).sum()*c*c/1e6
    assert abs(line/exact - 1) < .02, (line, exact)
    return line, exact, out


def job(seed):
    warnings.filterwarnings("ignore")
    from rl_search.scripts.check_multi import make_envs
    from rl_search.target import simulate
    get = load_runs()
    env = make_envs(1, seed)[0]
    tc = env.sampler.cfg.target
    ref = get(seed, "greedy", 1)
    track, exit_step = simulate(env.sampler.terr, tc, np.array(ref["lkp"]), ref["heading"], ref["speed_kmh"]/3.6,
                                np.random.default_rng(seed + 10**6))
    env.scen = SimpleNamespace(track=track, exit_step=exit_step, save_dt_s=tc.save_dt_s, speed_mps=ref["speed_kmh"]/3.6)
    c, lo, n, R = env.cfg.cell_m, env.lo, env.n, env.lrc_R
    pts = np.asarray(track[int(T_REGION/tc.save_dt_s)], float)
    ii = np.floor((pts - lo)/c).astype(int)
    env.cfg.margin_rate = 0.0

    def build(level):
        env.cfg.hdr_level, env._zone_key = level, None
        env._zone(T_REGION)
        z, iy0, ix0 = env._zone25
        out = np.zeros((n, n), bool)
        out[iy0:iy0 + z.shape[0], ix0:ix0 + z.shape[1]] = z
        return out, float(out[ii[:, 1], ii[:, 0]].mean())

    # smoothing spreads the mass, so _zone's level 0.85 holds about 89% of the particles: find the level that holds 85%
    a, b = .5, .85
    for _ in range(12):
        level = (a + b)/2
        region, held = build(level)
        a, b = (level, b) if held < TARGET else (a, level)
    region, held = build(b)
    area = float(region.sum()*c*c/1e6)
    full_swath = env.v*3600.0*2*R/1e6
    rows, maps = [], {}
    for m in METHODS:
        for N in NS:
            r = get(seed, m, N)
            first = np.full((n, n), np.inf, np.float32)
            for xy, tt in zip(r["xy"], r["tt"]):
                sweep(first, lo, c, np.array(xy, float), np.array(tt, float), R)
            done = np.isfinite(first)
            row = dict(seed=seed, method=m, n=N, region_km2=area, particles_in_region=held, hold=r["hold"],
                       ratio=float((done & region).sum()/region.sum()), swept_km2=float(done.sum()*c*c/1e6),
                       swath_max_km2=N*full_swath, heading=ref["heading"])
            for mm in MINUTES:
                row[f"ratio{mm}"] = float(((first <= T_H + mm*60) & region).sum()/region.sum())
            rows.append(row)
            if seed in (402, 404, 408):
                maps[f"{m}_{N}"] = done
    if maps:
        np.savez_compressed(HERE/f"hdr_maps_s{seed}.npz", region=region, lo=lo, c=c, **maps)
    print(f"seed {seed}: region {area:.1f} km2, particles inside {held:.3f}", flush=True)
    return rows


def main():
    print("sweep check (line, exact, circle by sample step):", check_sweep(), flush=True)
    with Pool(8) as pool:
        rows = [r for part in pool.imap_unordered(job, SEEDS, chunksize=1) for r in part]
    rows.sort(key=lambda r: (r["seed"], r["method"], r["n"]))
    (HERE/"hdr_coverage.json").write_text(json.dumps(rows))

    def sel(**kw):
        return [r for r in rows if all(r[k] == v for k, v in kw.items())]

    print("\nregion: mean area", np.mean([r["region_km2"] for r in sel(method="greedy", n=1)]).round(1), "km2, range",
          np.round([min(r["region_km2"] for r in rows), max(r["region_km2"] for r in rows)], 1),
          ", particles inside, mean", np.mean([r["particles_in_region"] for r in sel(method="greedy", n=1)]).round(3))
    print("\nN | SAROPS ratio, greedy ratio, diff (se, greedy higher, sarops higher) | >=95% count S/G | swept km2 S/G (share of max) | ratio at 15/30/45/60 S ; G")
    for N in NS:
        a, b = sel(method="sarops", n=N), sel(method="greedy", n=N)
        x, y = np.array([r["ratio"] for r in a]), np.array([r["ratio"] for r in b])
        d = y - x
        sx, sy = np.array([r["swept_km2"] for r in a]), np.array([r["swept_km2"] for r in b])
        mx = a[0]["swath_max_km2"]
        print(f"{N} | {x.mean():.3f} {y.mean():.3f} {d.mean():+.3f} (se {d.std(ddof=1)/np.sqrt(10):.3f}, {int((d > 0).sum())}, {int((d < 0).sum())}) | "
              f"{int((x >= .95).sum())}/{int((y >= .95).sum())} | {sx.mean():.1f} {sy.mean():.1f} ({sx.mean()/mx:.2f} {sy.mean()/mx:.2f}) | "
              + " ".join(f"{np.mean([r[f'ratio{m}'] for r in a]):.3f}" for m in MINUTES) + " ; "
              + " ".join(f"{np.mean([r[f'ratio{m}'] for r in b]):.3f}" for m in MINUTES))
    print("\nseed | region km2 | particles in | 1: S G | 3: S G | 5: S G | POS 1: S G | POS 5: S G")
    for s in SEEDS:
        g = {(r["method"], r["n"]): r for r in sel(seed=s)}
        print(s, f"{g[('greedy', 1)]['region_km2']:.1f} {g[('greedy', 1)]['particles_in_region']:.3f} | "
              + " | ".join(f"{g[('sarops', N)]['ratio']:.3f} {g[('greedy', N)]['ratio']:.3f}" for N in NS)
              + f" | {g[('sarops', 1)]['hold']:.3f} {g[('greedy', 1)]['hold']:.3f} | {g[('sarops', 5)]['hold']:.3f} {g[('greedy', 5)]['hold']:.3f}")
    print("\ncorrelation over scenarios of the two gaps (greedy - SAROPS): ratio vs POS")
    for N in NS:
        dr = [a["ratio"] - b["ratio"] for a, b in zip(sel(method="greedy", n=N), sel(method="sarops", n=N))]
        dp = [a["hold"] - b["hold"] for a, b in zip(sel(method="greedy", n=N), sel(method="sarops", n=N))]
        print(N, np.corrcoef(dr, dp)[0, 1].round(2))


if __name__ == "__main__":
    main()

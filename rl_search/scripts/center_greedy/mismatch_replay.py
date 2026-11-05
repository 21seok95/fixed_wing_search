"""Exploratory analysis, no planning rerun: the saved flights (planned on the nominal target model) scored against target
models they were not planned for, plus how much of each flight is spent in closed loops."""
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, REPO)
os.chdir(REPO)

import json
import warnings
from dataclasses import replace
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(REPO)/"rl_search/data/center_greedy"
SEEDS = list(range(400, 410))
METHODS = ("sarops", "greedy")
NS = (1, 3, 5)
T_H = 120.0
# name: (heading offset deg, speed factor, TargetConfig overrides)
TRUTHS = {
    "nominal": (0, 1, {}),
    "heading +45": (45, 1, {}),
    "heading -45": (-45, 1, {}),
    "heading +90": (90, 1, {}),
    "heading -90": (-90, 1, {}),
    "heading 180": (180, 1, {}),
    "heading unknown": (0, 1, dict(heading_std_deg=1000.0)),
    "speed x0.5": (0, .5, {}),
    "speed x1.5": (0, 1.5, dict(speed_cap_mps=12.5)),
    "LKP error 1 km": (0, 1, dict(lkp_pos_std_m=1000.0)),
    "mostly hiding": (0, 1, dict(p_move=.25, p_reorient=.15, p_stop=.6)),
    "always moving": (0, 1, dict(p_move=.9, p_reorient=.1, p_stop=0.0)),
}


def load_runs():
    new = {(r["seed"], r["method"], r["n"], r["sep"]): r for r in map(json.loads, open(HERE/"center_greedy.jsonl"))}
    old = {(r["seed"], r["mode"]): r for r in json.load(open("rl_search/data/compare_center/results.json"))}

    def get(seed, method, n):
        if method == "sarops" and n in (1, 5):
            r = old[(seed, "1" if n == 1 else "5 no sep")]
            return dict(r, hold=r["hold_pos"])
        return new[(seed, method, n, False)]
    return get


def loop_share(xy, window=9, thresh_deg=300.0):
    """Share of the flight (after the 120 s transit, 10 s samples) inside a closed loop: the signed heading change over
    90 s reaches 300 deg. At the 6 deg/s turn limit a full circle takes 60 s."""
    xy = np.asarray(xy, float)[12:]
    d = np.diff(xy, axis=0)
    h = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
    dh = np.diff(h)
    mark = np.zeros(len(h), bool)
    for i in range(len(dh) - window + 1):
        if abs(dh[i:i + window].sum()) >= np.deg2rad(thresh_deg):
            mark[i:i + window + 1] = True
    return float(mark.mean())


def check_loop_share():
    t = np.arange(0, 1200, 10.0)
    circle = np.stack([143*np.cos(t*np.deg2rad(6)), 143*np.sin(t*np.deg2rad(6))], 1)
    lanes, x, y, s = [], 0.0, 0.0, 1
    for _ in range(12):                     # lawnmower: 1 km lanes joined by opposite half circles
        for _ in range(7):
            lanes.append((x, y))
            x += s*150
        for a in np.deg2rad(np.arange(0, 180, 60)):
            lanes.append((x + s*143*np.sin(a), y + 143*(1 - np.cos(a))))
        y += 286
        s = -s
    a, b = loop_share(circle), loop_share(np.array(lanes))
    assert a > .95 and b < .01, (a, b)
    return a, b


def replay(env, scen, paths, times, cuts, pod=None):
    """POS at the cut times of the saved paths against ``scen`` (CPA + LRC, one pass state per UAV)."""
    from rl_search.mission import ParticleBelief
    if pod is not None:
        env.lrc_curve = lambda d: np.where(d <= env.lrc_R, pod, 0.0)
    surv = []
    for xy, tt in zip(paths, times):
        env.belief = ParticleBelief(scen["track"], scen["dt"], scen["exit"], True)
        env._pass_p, env._last_in = np.zeros(env.belief.n), np.full(env.belief.n, -np.inf)
        seq, k = [], 0
        for c in cuts:
            while k + 1 < len(tt) and tt[k + 1] <= c:
                env._detect_lrc(xy[k:k + 2], tt[k:k + 2])
                k += 1
            seq.append(env.belief.w.copy())
        surv.append(seq)
    if pod is not None:
        del env.lrc_curve
    return [float(1 - np.prod([s[i] for s in surv], axis=0).mean()) for i in range(len(cuts))]


def job(seed):
    warnings.filterwarnings("ignore")
    from rl_search.scripts.check_multi import make_envs
    from rl_search.target import simulate
    get = load_runs()
    env = make_envs(1, seed)[0]
    terr, tcfg = env.sampler.terr, env.sampler.cfg.target
    ref = get(seed, "greedy", 1)
    lkp, heading, speed = np.array(ref["lkp"]), ref["heading"], ref["speed_kmh"]/3.6
    runs = {(m, n): get(seed, m, n) for m in METHODS for n in NS}
    paths = {k: ([np.array(x, float) for x in r["xy"]], [np.array(t, float) for t in r["tt"]]) for k, r in runs.items()}
    rows = []
    cuts = [T_H + 1800.0, 1e9]
    for i, (name, (dh, sf, over)) in enumerate(TRUTHS.items()):
        rng = np.random.default_rng(seed + 10**6 + 1000*i)     # i = 0: the same particles as the held-out set
        track, exit_step = simulate(terr, replace(tcfg, **over), lkp, heading + dh, speed*sf, rng)
        scen = dict(track=track, exit=exit_step, dt=tcfg.save_dt_s)
        esc = float((exit_step >= 0).mean())
        for (m, n), (xy, tt) in paths.items():
            p30, pend = replay(env, scen, xy, tt, cuts)
            row = dict(seed=seed, truth=name, method=m, n=n, p30=p30, pend=pend, esc=esc, recorded=runs[(m, n)]["hold"])
            if i == 0:
                row["ever"] = replay(env, scen, xy, tt, [1e9], pod=1.0)[0]
                row["loop"] = float(np.mean([loop_share(x) for x in xy]))
            rows.append(row)
    print("seed", seed, "done", flush=True)
    return rows


def summarize(rows):
    def sel(**kw):
        return [r for r in rows if all(r[k] == v for k, v in kw.items())]

    print("\n--- replay check: nominal replay vs recorded held-out POS ---")
    d = [r["pend"] - r["recorded"] for r in sel(truth="nominal")]
    print(f"mean diff {np.mean(d):+.4f}, largest |diff| {np.max(np.abs(d)):.4f}")

    print("\n--- flight in closed loops; coverage (share of particles ever in range) ---")
    for n in NS:
        for m in METHODS:
            g = sel(truth="nominal", method=m, n=n)
            pos, ever = np.mean([r["pend"] for r in g]), np.mean([r["ever"] for r in g])
            print(f"N={n} {m:7s} loop share {np.mean([r['loop'] for r in g]):.3f}  POS {pos:.3f}  ever in range {ever:.3f}  "
                  f"0.7*ever {0.7*ever:.3f}  POS gained from repeat looks {pos - 0.7*ever:+.3f}")
        dd = [a["ever"] - b["ever"] for a, b in zip(sel(truth="nominal", method="greedy", n=n), sel(truth="nominal", method="sarops", n=n))]
        print(f"   ever in range, greedy - SAROPS: {np.mean(dd):+.3f} se {np.std(dd, ddof=1)/np.sqrt(len(dd)):.3f}, greedy higher {sum(x > 0 for x in dd)}/10")

    print("\n--- POS at end of flight per truth: SAROPS / greedy / diff (se, greedy wins) ---")
    for n in NS:
        print(f"N={n}")
        for t in TRUTHS:
            a, b = sel(truth=t, method="sarops", n=n), sel(truth=t, method="greedy", n=n)
            x, y = np.array([r["pend"] for r in a]), np.array([r["pend"] for r in b])
            d = y - x
            print(f"  {t:16s} {x.mean():.3f} {y.mean():.3f} {d.mean():+.3f} (se {d.std(ddof=1)/np.sqrt(len(d)):.3f}, {int((d > 0).sum())}/10)  esc {np.mean([r['esc'] for r in a]):.2f}")

    print("\n--- across all mismatched truths (mean per scenario, then paired over scenarios) ---")
    bad = [t for t in TRUTHS if t != "nominal"]
    for n in NS:
        for key, label in (("pend", "end"), ("p30", "30 min")):
            S = {m: np.array([[sel(seed=s, truth=t, method=m, n=n)[0][key] for t in bad] for s in SEEDS]) for m in METHODS}
            N0 = {m: np.array([sel(seed=s, truth="nominal", method=m, n=n)[0][key] for s in SEEDS]) for m in METHODS}
            d = S["greedy"].mean(1) - S["sarops"].mean(1)
            w = S["greedy"].min(1) - S["sarops"].min(1)
            print(f"N={n} {label:6s} nominal S {N0['sarops'].mean():.3f} G {N0['greedy'].mean():.3f} | mismatched mean S {S['sarops'].mean():.3f} "
                  f"G {S['greedy'].mean():.3f} diff {d.mean():+.3f} (se {d.std(ddof=1)/np.sqrt(10):.3f}, {int((d > 0).sum())}/10) | worst truth S "
                  f"{S['sarops'].min(1).mean():.3f} G {S['greedy'].min(1).mean():.3f} diff {w.mean():+.3f} (se {w.std(ddof=1)/np.sqrt(10):.3f}, {int((w > 0).sum())}/10) | "
                  f"loss vs nominal S {N0['sarops'].mean() - S['sarops'].mean():.3f} G {N0['greedy'].mean() - S['greedy'].mean():.3f}")


def main():
    print("loop metric check (circle, lawnmower):", check_loop_share(), flush=True)
    with Pool(8) as pool:
        rows = [r for part in pool.imap_unordered(job, SEEDS, chunksize=1) for r in part]
    (HERE/"mismatch_replay.json").write_text(json.dumps(rows))
    summarize(rows)


if __name__ == "__main__":
    main()

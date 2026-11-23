"""Fingerprint of a training-scenario pool (user 2026-11-23: the Ubuntu PC rebuilds the Chungju 120 km pool -- 4,000 scenarios, 63 GB, not in
git -- with the same seeds, and must get the same scenarios as the Mac). For a sample of the pool's files (every ``step``-th seed) it keeps what
defines a scenario: launch point and heading, LKP, the evader's heading and speed, and the particle tracks' centre and spread at a few times
plus the number of particles that leave. ``save`` writes <region>/<pool>_fingerprint.json (small, added to git with -f); ``check`` rebuilds
the numbers from the local pool and compares them (positions within ``--tol`` m).

    python -m rl_search.scripts.rl_eval.pool_fingerprint save  --region chungju120
    python -m rl_search.scripts.rl_eval.pool_fingerprint check --region chungju120
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

REAL = Path("rl_search/data/real")
TIMES = (0, 30, 60, 120, 240, 420)          # track rows (save_dt_s 10 s: 0, 5, 10, 20, 40, 70 min)


def _numbers(path: Path) -> dict:
    d = pickle.load(open(path, "rb"))
    sc = d["scen"]
    tr = np.asarray(sc.track, float)
    rows = [t for t in TIMES if t < len(tr)]
    return dict(seed=int(d["seed"]), type=str(d["type"]), start=np.round(sc.start[:2], 2).tolist(), lkp=np.round(sc.lkp, 2).tolist(),
                start_heading_deg=round(float(sc.start_heading_deg), 4), heading_deg=round(float(sc.heading_deg), 4),
                speed_mps=round(float(sc.speed_mps), 4), n_track=int(len(tr)), n_particles=int(tr.shape[1]),
                exits=int(np.sum(np.asarray(sc.exit_step) >= 0)),
                centre={str(t): np.round(np.nanmean(tr[t], 0), 2).tolist() for t in rows},
                spread={str(t): round(float(np.nanmean(np.nanstd(tr[t], 0))), 2) for t in rows})


def _files(region: str, pool: str, step: int) -> list:
    files = sorted((REAL/region/pool).glob("*_s*.pkl"), key=lambda p: int(p.stem.split("_s")[-1]))
    return files[::step]


def save(region: str, pool: str, step: int):
    files = _files(region, pool, step)
    out = dict(region=region, pool=pool, step=step, n_files=len(list((REAL/region/pool).glob("*_s*.pkl"))),
               samples=[_numbers(p) for p in files])
    path = REAL/region/f"{pool}_fingerprint.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"{path}: {len(files)} of {out['n_files']} scenarios")


def check(region: str, pool: str, tol: float) -> bool:
    ref = json.loads((REAL/region/f"{pool}_fingerprint.json").read_text())
    n_here = len(list((REAL/region/pool).glob("*_s*.pkl")))
    ok = n_here >= ref["n_files"]
    print(f"{region}/{pool}: {n_here} scenario files here, {ref['n_files']} on the Mac -> {'OK' if ok else 'MISSING FILES'}")
    worst, bad = 0.0, []
    for r in ref["samples"]:
        p = REAL/region/pool/f"{r['type']}_s{r['seed']}.pkl"
        if not p.exists():
            bad.append(f"seed {r['seed']}: missing")
            continue
        h = _numbers(p)
        for key in ("type", "n_track", "n_particles", "exits"):
            if h[key] != r[key]:
                bad.append(f"seed {r['seed']}: {key} {h[key]} != {r[key]}")
        d = [np.abs(np.subtract(h[k], r[k])).max() for k in ("start", "lkp")]
        d += [abs(h[k] - r[k]) for k in ("start_heading_deg", "heading_deg", "speed_mps")]
        d += [np.abs(np.subtract(h["centre"][t], r["centre"][t])).max() for t in r["centre"]]
        d += [abs(h["spread"][t] - r["spread"][t]) for t in r["spread"]]
        worst = max(worst, float(max(d)))
        if max(d) > tol:
            bad.append(f"seed {r['seed']}: largest difference {max(d):.3f}")
    ok = ok and not bad
    print(f"{len(ref['samples'])} sampled scenarios, largest difference {worst:.3g} (tolerance {tol:g}) -> {'PASS' if ok else 'FAIL'}")
    for b in bad[:20]:
        print("  ", b)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("save", "check"))
    ap.add_argument("--region", required=True)
    ap.add_argument("--pool", default="rl_pool_v2")
    ap.add_argument("--step", type=int, default=100, help="save: every step-th seed (4,000 files -> 40)")
    ap.add_argument("--tol", type=float, default=1.0, help="check: largest allowed difference (m, deg, m/s)")
    a = ap.parse_args()
    if a.what == "save":
        save(a.region, a.pool, a.step)
    else:
        sys.exit(0 if check(a.region, a.pool, a.tol) else 1)


if __name__ == "__main__":
    main()

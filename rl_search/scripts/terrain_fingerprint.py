"""Is a terrain rebuilt on another machine the same as the Mac's? (user 2026-11-18: the slim local-PC branch keeps only the OSM downloads,
gzipped; the DEM and the land cover are fixed releases the local PC downloads again, so the rebuilt maps must be checked.)

``save`` writes rl_search/data/real/<region>/terrain_fingerprint.json on the Mac: per array of the terrain file, the road graph and the
land cover, a value fingerprint (sha1 of the values as int64 / float64 / uint8) and summary values (mean, standard deviation, a few
quantiles, true share for masks). ``check`` rebuilds nothing; it compares this machine's files with the saved ones: "same" when every
fingerprint matches; otherwise per array how far the summaries are apart, so a floating-point difference between machines (tiny) can be
told from a different download (OSM changed: other roads) or a broken file.

    python -m rl_search.scripts.terrain_fingerprint save --region amboy mojave chungju      # on the Mac, committed with the code
    python -m rl_search.scripts.terrain_fingerprint check --region amboy mojave chungju     # on the local PC after make_real_terrain
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

DATA = Path("rl_search/data/real")


def files(region: str, cell_m: float = 25.0) -> dict[str, Path]:
    from rl_search.real_terrain import terrain_file
    sfx = "" if cell_m == 50 else f"_{cell_m:g}m"
    return {"terrain": terrain_file(region, cell_m), "road_graph": DATA/region/f"road_graph{sfx}.npz",
            "landcover": DATA/region/f"landcover{sfx}.npz"}


def summary(a: np.ndarray) -> dict:
    a = np.asarray(a)
    if a.dtype.kind in "USV" or a.dtype.names:                 # strings and structured arrays: fingerprint only
        b = a.tobytes()
        return dict(sha=hashlib.sha1(b).hexdigest()[:16], shape=list(a.shape), dtype=a.dtype.str)
    b = a.astype(np.uint8) if a.dtype == bool else a.astype(np.int64) if np.issubdtype(a.dtype, np.integer) else a.astype(np.float64)
    out = dict(sha=hashlib.sha1(np.ascontiguousarray(b).tobytes()).hexdigest()[:16], shape=list(a.shape), dtype=a.dtype.str)
    f = b.astype(np.float64).ravel()
    f = f[np.isfinite(f)]
    if len(f):
        out.update(mean=float(f.mean()), std=float(f.std()), q=[float(x) for x in np.quantile(f, [0, .01, .5, .99, 1])])
    return out


def take(region: str) -> dict:
    out = {}
    for kind, p in files(region).items():
        if not p.exists():
            out[kind] = None
            continue
        d = np.load(p, allow_pickle=False)
        out[kind] = {k: summary(d[k]) for k in d.files if k != "config"}
    return out


def save(regions):
    for r in regions:
        fp = take(r)
        missing = [k for k, v in fp.items() if v is None]
        if missing:
            sys.exit(f"{r}: missing {missing}; build it first (make_real_terrain --region {r} --cell 25)")
        (DATA/r/"terrain_fingerprint.json").write_text(json.dumps(fp, indent=1))
        print(f"{r}: {sum(len(v) for v in fp.values())} arrays saved")


def check(regions) -> bool:
    ok_all = True
    for r in regions:
        ref_p = DATA/r/"terrain_fingerprint.json"
        if not ref_p.exists():
            print(f"{r}: no saved fingerprint ({ref_p})")
            ok_all = False
            continue
        ref, now = json.loads(ref_p.read_text()), take(r)
        bad = []
        for kind, arrs in ref.items():
            if now.get(kind) is None:
                bad.append(f"{kind}: file missing ({files(r)[kind]})")
                continue
            for k, s in arrs.items():
                n = now[kind].get(k)
                if n is None:
                    bad.append(f"{kind}.{k}: missing")
                elif n["shape"] != s["shape"]:
                    bad.append(f"{kind}.{k}: shape {n['shape']} (Mac {s['shape']})")
                elif n["sha"] != s["sha"]:
                    gap = ""
                    if "mean" in s and "mean" in n:
                        gap = f", mean {n['mean']:.6g} (Mac {s['mean']:.6g}), std {n['std']:.6g} (Mac {s['std']:.6g})"
                    bad.append(f"{kind}.{k}: values differ{gap}")
        if bad:
            ok_all = False
            print(f"{r}: DIFFERENT from the Mac in {len(bad)} arrays")
            for b in bad:
                print("   ", b)
        else:
            print(f"{r}: same as the Mac ({sum(len(v) for v in ref.values())} arrays)")
    return ok_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("save", "check"))
    ap.add_argument("--region", nargs="+", required=True)
    a = ap.parse_args()
    if a.what == "save":
        save(a.region)
    else:
        sys.exit(0 if check(a.region) else 1)


if __name__ == "__main__":
    main()

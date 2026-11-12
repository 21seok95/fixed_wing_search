"""The re-scored test flights (rescore_area.py) with the real-time window A (``area_score.RealtimeArea``: the 5 min windows become 10 s
frames; user, 2026-11-11). The camera images of each flight come from its web run (1 s ground quads), so nothing is flown again; they are
swept over the hold-out particles' real-time region. Adds A_rt, total_rt and the share of A_rt gained while turning to the rows of
rescore_area_test.json and writes rescore/<method>_k<k>_rt.npz (gain per sample).

    python -m rl_search.scripts.rl_eval.rescore_realtime --workers 2
"""
from __future__ import annotations

import argparse
import base64
import gzip
import json
import time
import warnings
from multiprocessing import get_context

import numpy as np

from rl_search.scripts.rl_eval.rescore_area import OUT, TURN_DPS, X_OLD

RUN_ID = {"rl": "eval-test-s3-k{k}-rl-6200000", "sarops": "eval-test-s3-k{k}-sarops"}


def run_id(method: str, k: int) -> str:
    return RUN_ID[method].format(k=k) if method in RUN_ID else f"eval-test-s3-k{k}-rl-{method}"


def job(args):
    method, k, x = args[:3]
    quality, tau = (args[3], args[4]) if len(args) > 3 else (False, 0.0)
    warnings.filterwarnings("ignore")
    from rl_search.area_score import RealtimeArea
    from rl_search.detection import DetectionConfig
    from rl_search.real_scenario import VehicleScenarios
    from rl_search.rl_eval_set import REGION, make_case, vehicle_config
    from rl_search.rl_real import preload
    from rl_search.scripts.real_map.vehicle_web import RUNS
    t0 = time.perf_counter()
    st = preload(REGION)
    vs = VehicleScenarios(st["sampler"], vehicle_config(), region=REGION, net=st["net"])
    case, scen, vres, hold = make_case(vs, "test", 3, k)
    z0 = np.load(OUT/f"{method}_k{k}.npz")
    if "quad" in z0.files:                                          # flights re-scored since 2026-11-11 keep their image quads
        lt, quad = z0["log"][:, 0].astype(float), z0["quad"].astype(float)
    else:                                                           # older ones: from the web run of the same flight
        d = json.load(gzip.open(RUNS/run_id(method, k)/"run.json.gz", "rt"))
        U = d["opt"]["uavs"][0]
        lt = np.frombuffer(base64.b64decode(U["log_t"]), np.float32).astype(float)
        quad = np.frombuffer(base64.b64decode(U["quad"]), np.float32).reshape(-1, 4, 2).astype(float)
    ra = RealtimeArea(st["terr"], 150.0, hold.track, hold.exit_step, hold.save_dt_s, x, quality=quality)
    z = np.load(OUT/f"{method}_k{k}.npz")
    log = z["log"].astype(float)                                    # hold-out replay: t, x, y, yaw, roll per 1 s sample
    pos = np.clip(np.searchsorted(log[:, 0], lt - 1e-3), 0, len(log) - 1)
    om = np.degrees(9.80665*np.tan(log[pos, 4])/33.0)
    matched = np.abs(log[pos, 0] - lt) < .01
    q = DetectionConfig(rate_tau_dps=tau).rate_factor(om) if quality else np.ones(len(lt))   # turn rate from the roll (no gimbal rate here)
    o = np.argsort(lt, kind="stable")
    g = np.zeros(len(lt))
    for i in o:
        g[i] = ra.mark_quad(quad[i], lt[i], q[i])
    turning = matched & (np.abs(om) > TURN_DPS)
    sfx = "_rt" if not quality else f"_rtq{tau:g}"
    np.savez_compressed(OUT/f"{method}_k{k}{sfx}.npz", t=lt.astype(np.float32), gain=g, curve=ra.curve(), size=ra.size)
    return dict(method=method, k=k, A_rt=ra.A, gained_turning_rt=float(g[turning].sum()/max(g.sum(), 1e-12)), matched=float(matched.mean()),
                region_km2_rt=float(ra.size.mean()*25.0**2/1e6), sec=round(time.perf_counter() - t0, 1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--methods", nargs="*", default=["sarops", "rl"])
    ap.add_argument("--x", type=float, default=X_OLD)
    ap.add_argument("--quality", action="store_true", help="A4: weight each sweep by the turn-rate factor (needs --blur-tau for D2)")
    ap.add_argument("--blur-tau", type=float, default=0.0, help="0 = the linear factor (D1), > 0 = exp(-w/tau) (D2)")
    args = ap.parse_args()
    path = OUT.parent/"rescore_area_test.json"
    rows = json.loads(path.read_text())
    todo = [(r["method"], r["k"], args.x, args.quality, args.blur_tau) for r in rows if r["method"] in args.methods]
    key = "A_rt" if not args.quality else f"A_rtq{args.blur_tau:g}"
    with get_context("spawn").Pool(args.workers) as pool:
        res = {(r["method"], r["k"]): r for r in pool.imap_unordered(job, todo)}
    for r in rows:
        u = res.get((r["method"], r["k"]))
        if u:
            r[key], r["total" + key[1:]] = u["A_rt"], u["A_rt"] + r["pos"]
            r.setdefault("gained_turning", {})[key] = u["gained_turning_rt"]
            r["region_km2_rt"] = u["region_km2_rt"]
            print(f"{r['method']:6s} k{r['k']:02d}: A 5 min {r['A_old']:.3f} -> real time {u['A_rt']:.3f}, while turning "
                  f"{r['gained_turning']['A_old']:.0%} -> {u['gained_turning_rt']:.0%} (samples matched {u['matched']:.0%}), {u['sec']:.0f} s")
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    print("saved", path)


if __name__ == "__main__":
    main()

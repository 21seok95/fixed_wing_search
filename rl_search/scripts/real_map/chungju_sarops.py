"""Step 2 check: SAROPS (accordion + local search, task mode) on Chungju with vehicle particles and the oblique camera.

One UAV, one scenario, three lane variants:
  plain    lanes as before (the planner treats the footprint as if it were under the UAV)
  lead     the UAV flies each lane 510 m behind, so the footprint centre runs over the rectangle's lane
  prelook  lead + the gimbal points at the next lane's start from 8 s before a lane end through the lane change

Figure: rl_search/data/real/chungju/sarops_step2_<seed>.png, results in .json next to it.

    python -m rl_search.scripts.real_map.chungju_sarops --seed 1
"""
from __future__ import annotations

import argparse
import json
import time
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.patches import Polygon

from rl_search.real_terrain import DATA

plt.rcParams["font.family"] = ["AppleGothic", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
VARIANTS = {"plain": (0.0, 0.0, "기존 레인"), "lead": (510.0, 0.0, "발자국 앞섬 반영"),
            "prelook": (510.0, 8.0, "앞섬 + U턴 짐벌 8초 앞당김")}
COLORS = {"plain": "#7f8c8d", "lead": "#2a4d7a", "prelook": "#c0392b"}


def run(seed, variant, particles=2000, scen=None):
    from rl_search.multi import run_team_launch
    from rl_search.real_scenario import VehicleScenarios, make_env
    from rl_search.sarops import SaropsPlanner
    lead, pre, _ = VARIANTS[variant]
    env = make_env(seed, lead_m=lead, prelook_s=pre)
    if scen is None:
        scen = VehicleScenarios(env.sampler).make(seed, particles)
    env.sample_log = []
    pol = SaropsPlanner(W=133.0, n_starts=3, big_jump=True, search_s=600.0, max_transit_s=600.0, lane_mode="cover",
                        peak_timing="intercept")
    pol.team = [pol]
    t0 = time.perf_counter()
    res = run_team_launch([env], [pol], dict(scenario=scen, stage=3), 60.0)
    log = np.array(env.sample_log)
    rects = [(tp, r.cx, r.cy, r.ell, r.wid, r.theta) for tp, r, _ in pol.rects]
    return dict(variant=variant, pos_t=np.asarray(res["t"]).tolist(), pos=np.asarray(res["pos"]).tolist(),
                xy=np.asarray(res["xy"][0]).tolist(), tt=np.asarray(res["tt"][0]).tolist(), rects=rects,
                log=log, sec=time.perf_counter() - t0, final=float(res["pos"][-1]), scen=scen,
                tasks=[dict(t0=k["t0"], transit=k["transit"], lane_t=k["lane_t"], end=k["end"]) for k in pol.tasks])


def footprints(log, every=5):
    from rl_search.detection import DetectionConfig
    cfg = DetectionConfig()
    th, tv = np.tan(np.radians(cfg.hfov_deg/2)), np.tan(np.radians(cfg.vfov_deg/2))
    polys = []
    for row in log[::every]:
        x, y = row[1], row[2]
        f, r, u = row[5:8], row[8:11], row[11:14]
        q = []
        for sh, sv in ((-1, -1), (-1, 1), (1, 1), (1, -1)):
            d = f + sh*th*r + sv*tv*u
            t = cfg.alt_m/max(-d[2], 1e-3)
            q.append((x + t*d[0], y + t*d[1]))
        polys.append(np.array(q)/1000)
    return polys


def rect_poly(cx, cy, ell, wid, theta):
    c, s = np.cos(theta), np.sin(theta)
    pts = np.array([[-ell/2, -wid/2], [ell/2, -wid/2], [ell/2, wid/2], [-ell/2, wid/2]])
    return (pts @ np.array([[c, s], [-s, c]]) + [cx, cy])/1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--particles", type=int, default=2000)
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS))
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    from rl_search.real_scenario import VehicleScenarios, make_env
    from rl_search.scripts.real_map.vehicle_tuner import map_image
    probe = make_env(args.seed)
    vs = VehicleScenarios(probe.sampler)
    scen = vs.make(args.seed, args.particles)
    out = {}
    for v in args.variants:
        out[v] = run(args.seed, v, args.particles, scen)
        o = out[v]          # flight log for the zoomed comparison (chungju_zoom.py)
        np.savez_compressed(DATA/"chungju"/f"sarops_step2_{args.seed}_{v}.npz", log=o["log"], xy=np.array(o["xy"]),
                            tt=np.array(o["tt"]), rects=np.array(o["rects"]), pos_t=np.array(o["pos_t"]),
                            pos=np.array(o["pos"]), tasks=np.array([[k["t0"], k["transit"], k["lane_t"],
                                                                     np.nan if k["end"] is None else k["end"]]
                                                                    for k in o["tasks"]]), particles=args.particles)
        print(f"{v}: POS {out[v]['final']:.3f}, plans {len(out[v]['rects'])}, {out[v]['sec']:.0f} s", flush=True)

    img, ext = map_image(vs.net, 1600)
    nv = len(args.variants)
    fig = plt.figure(figsize=(7*nv + 1, 13), layout="constrained")
    gs = fig.add_gridspec(2, nv, height_ratios=[1.45, 1])
    track = scen.track
    for j, v in enumerate(args.variants):
        o = out[v]
        ax = fig.add_subplot(gs[0, j])
        ax.imshow(img, extent=ext, origin="upper", alpha=.6)
        xy = np.array(o["xy"])/1000
        ax.add_collection(PolyCollection(footprints(o["log"]), facecolors=COLORS[v], edgecolors="none", alpha=.12))
        k = min(int(round(1800/scen.save_dt_s)), len(track) - 1)
        ax.scatter(track[k, :, 0]/1000, track[k, :, 1]/1000, s=2, c="#f39c12", lw=0, alpha=.7, label="particle 30분")
        k = min(int(round(3600/scen.save_dt_s)), len(track) - 1)
        ax.scatter(track[k, :, 0]/1000, track[k, :, 1]/1000, s=2, c="#8e44ad", lw=0, alpha=.5, label="particle 60분")
        for tp, cx, cy, ell, wid, th in o["rects"]:
            ax.add_patch(Polygon(rect_poly(cx, cy, ell, wid, th), fill=False, ec="black", lw=1.2))
            ax.text(cx/1000, cy/1000, f"{tp/60:.0f}분", fontsize=7, ha="center", va="center",
                    bbox=dict(fc="white", ec="none", alpha=.7, pad=.5))
        ax.plot(xy[:, 0], xy[:, 1], color=COLORS[v], lw=1.1)
        ax.plot(*(scen.lkp/1000), "*", ms=14, mfc="yellow", mec="black")
        ax.plot(*(scen.start/1000), "^", ms=10, color="black")
        allxy = np.vstack([np.array(out[w]["xy"])[np.array(out[w]["tt"]) >= out[w]["tasks"][0]["t0"] + out[w]["tasks"][0]["transit"]]/1000
                           for w in args.variants if out[w]["tasks"]])
        lo, hi = allxy.min(0) - 1.0, allxy.max(0) + 1.0
        span = max(hi - lo)
        mid = (lo + hi)/2
        ax.set_xlim(mid[0] - span/2, mid[0] + span/2); ax.set_ylim(mid[1] - span/2, mid[1] + span/2); ax.set_aspect("equal")
        ax.set_title(f"{VARIANTS[v][2]}\n최종 POS {o['final']:.3f}, 계획 {len(o['rects'])}번 (사각형 = 탐색 구역, 색 = 본 범위)", fontsize=11)
        ax.set_xlabel("동쪽 (km)"); ax.set_ylabel("북쪽 (km)")
        if j == 0:
            ax.legend(loc="lower left", fontsize=8, markerscale=4)
    ax = fig.add_subplot(gs[1, :max(1, nv - 1)])
    for v in args.variants:
        o = out[v]
        ax.plot(np.array(o["pos_t"])/60, o["pos"], color=COLORS[v], lw=2.2, label=f"{VARIANTS[v][2]}: {o['final']:.3f}")
    ax.set_xlabel("LKP 후 시간 (분)"); ax.set_ylabel("누적 POS"); ax.legend(); ax.set_title("누적 POS", fontsize=11)
    if nv > 1:
        ax = fig.add_subplot(gs[1, nv - 1])
        for v in args.variants:
            log = out[v]["log"]
            seen_gim = np.isfinite(log[:, 14]).mean() if len(log) else 0
            dep = np.degrees(np.arcsin(-log[:, 7])) if len(log) else []
            ax.hist(dep, bins=np.arange(10, 81, 2.5), histtype="step", lw=2, color=COLORS[v],
                    label=f"{VARIANTS[v][2]} (짐벌 지향 {seen_gim:.0%})")
        ax.set_xlabel("땅 기준 내려보는 각 (°)"); ax.set_ylabel("1초 표본 수"); ax.legend(fontsize=8)
        ax.set_title("시선 각도 분포", fontsize=11)
    m = scen.metrics
    fig.suptitle(f"충주 2단계 점검: 시나리오 {args.seed} (LKP {scen.lkp[0]/1000:.1f}, {scen.lkp[1]/1000:.1f} km, 방향 {scen.heading_deg:.0f}°, "
                 f"출발점까지 {m['start_lkp_km']:.1f} km), 차량 particle {args.particles}개, 무인기 1대 33 m/s, SAROPS 10분 작업", fontsize=13)
    path = DATA/"chungju"/f"sarops_step2_{args.seed}.png"
    fig.savefig(path, dpi=100)
    plt.close(fig)
    summary = {v: dict(final=o["final"], plans=len(o["rects"]), sec=round(o["sec"], 1), tasks=o["tasks"],
                       pos_10min=[float(np.interp(t*60, o["pos_t"], o["pos"])) for t in range(0, 61, 10)]) for v, o in out.items()}
    path.with_suffix(".json").write_text(json.dumps(dict(seed=args.seed, lkp=scen.lkp.tolist(), heading=scen.heading_deg,
                                                         metrics=m, results=summary), indent=1, ensure_ascii=False))
    print(f"Saved {path}")


if __name__ == "__main__":
    main()

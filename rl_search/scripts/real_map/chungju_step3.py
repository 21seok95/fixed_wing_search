"""Step 3 (first run): SAROPS on Chungju vehicle scenarios with the settings agreed on 2026-11-06, one UAV, 60 min.

Settings: lanes flown 510 m behind (footprint lead) + gimbal pointed at the next lane's start from 8 s before each
lane end; each pass finds 0.7 x the detection model's pass POD (pass_mode "scale"); planner POD 0.7
(lane width = camera footprint width 190 m; the first run used 173 m, kept in step3_173/); 30 deg bank turns (radius 192 m); LKP >= 15 km from the
map edge.

Lane time (default ``flown``, user 2026-11-06): the planner times each candidate's lanes by flying the follower on a
copy of the plant (``accordion.flown_lane_time``); ``geometry`` = ``fixed_lane_time`` (1.6x short here; step3/).
``--max-transit``: farthest rectangle (s of transit) the planner may pick (600 s so far).

Outputs in rl_search/data/real/chungju/<out>/ (default step3_<lane-time>[_tr<min>]): run_<seed>.npz, maps.png,
summary.png, summary.json

    python -m rl_search.scripts.real_map.chungju_step3 --seeds 1 2 3 4 5 6 7 8 9 10 --jobs 4
    python -m rl_search.scripts.real_map.chungju_step3 --lane-time geometry --out step3
    python -m rl_search.scripts.real_map.chungju_step3 --max-transit 1800
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from multiprocessing import get_context

import numpy as np

from rl_search.real_terrain import DATA

OUT = DATA/"chungju"/"step3"
LEAD_M, PRELOOK_S = 510.0, 8.0


def run(seed, particles=2000, lane_time="flown", out=OUT, max_transit_s=600.0, pass_gap=2.0, cap="now", vehicle=None):
    """``cap``: "now" (forced end at transit + 10 + 2 min), "A" (same, and the local search may not add passes beyond the
    10 min budget), "none" (no forced end). ``vehicle``: VehicleConfig overrides (road / off-road use)."""
    warnings.filterwarnings("ignore")
    from rl_search.multi import run_team_launch
    from rl_search.real_scenario import BANK_DEG, W_PLAN, VehicleScenarios, make_env
    from rl_search.sarops import SaropsPlanner
    from rl_search.vehicle import VehicleConfig
    t0 = time.perf_counter()
    env = make_env(seed, lead_m=LEAD_M, prelook_s=PRELOOK_S, detection=dict(pass_mode="scale", pass_gap_s=pass_gap),
                   bank_deg=BANK_DEG)
    env.lane_time_model = lane_time
    env.spacing_budget = cap == "A"
    scen = VehicleScenarios(env.sampler, VehicleConfig(**vehicle) if vehicle else None).make(seed, particles)
    env.sample_log = []
    pol = SaropsPlanner(W=W_PLAN, n_starts=3, big_jump=True, search_s=600.0, max_transit_s=max_transit_s, lane_mode="cover",
                        peak_timing="intercept", extra_s=7200.0 if cap == "none" else 120.0)
    pol.team = [pol]
    res = run_team_launch([env], [pol], dict(scenario=scen, stage=3), 60.0)
    found = 1 - env.surv
    rects = np.array([(tp, r.cx, r.cy, r.ell, r.wid, r.theta) for tp, r, _ in pol.rects])
    tasks = np.array([[k["t0"], k["transit"], np.nan if k["lane_t"] is None else k["lane_t"],
                       np.nan if k["end"] is None else k["end"], 1.0 if k.get("how") == "cap" else 0.0]
                      for k in pol.tasks])
    np.savez_compressed(out/f"run_{seed}.npz", log=np.array(env.sample_log), xy=res["xy"][0], tt=res["tt"][0],
                        pos_t=res["t"], pos=res["pos"], rects=rects, tasks=tasks, found=found,
                        lkp=scen.lkp, start=scen.start, heading=scen.heading_deg, particles=particles,
                        escaped=env.belief.escaped, off_share=scen.metrics["off_share"])
    sec = time.perf_counter() - t0
    print(f"seed {seed}: POS {res['pos'][-1]:.3f}, plans {len(rects)}, escaped {env.belief.escaped:.2f}, {sec:.0f} s",
          flush=True)
    return seed


def in_rect(px, py, r):
    _, cx, cy, ell, wid, th = r
    c, s = np.cos(th), np.sin(th)
    a = (px - cx)*c + (py - cy)*s
    b = -(px - cx)*s + (py - cy)*c
    return (np.abs(a) <= ell/2) & (np.abs(b) <= wid/2)


def stats(o):
    """Time split and where the camera looked."""
    log, tasks, rects = o["log"], o["tasks"], o["rects"]
    t = log[:, 0]
    T = t[-1] - t[0] + 1
    roll = np.abs(np.degrees(log[:, 4]))
    f = log[:, 5:8]
    tc = 500.0/np.maximum(-f[:, 2], 1e-3)
    fx, fy = log[:, 1] + tc*f[:, 0], log[:, 2] + tc*f[:, 1]
    on_lane = np.zeros(len(t), bool)
    fp_in = np.zeros(len(t), bool)
    for k, row in enumerate(tasks):
        t_l = row[2]
        t_e = row[3] if np.isfinite(row[3]) else t[-1]
        if not np.isfinite(t_l):
            continue
        m = (t >= t_l) & (t <= t_e)
        on_lane |= m
        fp_in[m] = in_rect(fx[m], fy[m], rects[k])
    turning = on_lane & (roll > 10)
    return dict(lane=on_lane.mean()*len(t)/T, turn_in_lane=turning.sum()/max(on_lane.sum(), 1),
                fp_in_rect=fp_in.sum()/max(on_lane.sum(), 1), capped=float(np.nansum(tasks[:, 4])/max(len(tasks), 1)),
                first_lane_min=float(tasks[0, 2]/60) if len(tasks) else np.nan)


def figures(seeds, particles, out=OUT, title=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.patches import Polygon
    from rl_search.real_scenario import VehicleScenarios, make_env
    from rl_search.scripts.real_map.chungju_sarops import rect_poly
    from rl_search.scripts.real_map.vehicle_tuner import map_image
    plt.rcParams["font.family"] = ["AppleGothic", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False
    warnings.filterwarnings("ignore")

    probe = make_env(seeds[0])
    vs = VehicleScenarios(probe.sampler)
    img, ext = map_image(vs.net, 2400)
    runs = {s: dict(np.load(out/f"run_{s}.npz")) for s in seeds}
    nc = 5
    nr = int(np.ceil(len(seeds)/nc))
    fig, axs = plt.subplots(nr, nc, figsize=(5.2*nc, 5.6*nr), layout="constrained", squeeze=False)
    for ax in axs.flat[len(seeds):]:
        ax.axis("off")
    for ax, s in zip(axs.flat, seeds):
        o = runs[s]
        scen = vs.make(s, particles)
        xy = o["xy"]/1000
        lo, hi = xy.min(0), xy.max(0)
        lo = np.minimum(lo, scen.lkp/1000) - 1.5; hi = np.maximum(hi, scen.lkp/1000) + 1.5
        span = max(hi - lo); mid = (lo + hi)/2
        win = (mid[0] - span/2, mid[0] + span/2, mid[1] - span/2, mid[1] + span/2)
        ax.imshow(img, extent=ext, origin="upper", alpha=.45)
        # particle density over the flight (every 5 min, 5..60), not weighted by the search
        P = np.concatenate([scen.track[min(int(round(m*60/scen.save_dt_s)), len(scen.track) - 1)]
                            for m in range(5, 61, 5)])
        cell = span/90
        bins = [np.arange(win[0], win[1] + cell, cell), np.arange(win[2], win[3] + cell, cell)]
        H, _, _ = np.histogram2d(P[:, 0]/1000, P[:, 1]/1000, bins=bins)
        H = H/H.sum()
        ax.imshow(np.where(H > 0, H, np.nan).T, origin="lower", extent=win, cmap="YlOrRd", norm=LogNorm(1e-4, 2e-2),
                  alpha=.85, interpolation="nearest")
        ax.plot(xy[:, 0], xy[:, 1], color="#1f4e8c", lw=1.0)
        for tp, cx, cy, ell, wid, th in o["rects"]:
            ax.add_patch(Polygon(rect_poly(cx, cy, ell, wid, th), fill=False, ec="black", lw=1.4))
            ax.text(cx/1000, cy/1000, f"{tp/60:.0f}", fontsize=7, ha="center", va="center",
                    bbox=dict(fc="white", ec="none", alpha=.75, pad=.4))
        ax.plot(*(scen.lkp/1000), "*", ms=14, mfc="yellow", mec="black")
        ax.set_xlim(win[0], win[1]); ax.set_ylim(win[2], win[3]); ax.set_aspect("equal")
        ax.set_title(f"시나리오 {s}: POS {o['pos'][-1]:.3f}, 지도 밖 {float(o['escaped']):.2f}, 계획 {len(o['rects'])}번",
                     fontsize=10)
        ax.tick_params(labelsize=7)
    lt = ("레인 시간 = 모의 비행" if "flown" in out.name else "레인 시간 = 기하 근사") + \
        (f", 이동 제한 {out.name.split('_tr')[1]}분" if "_tr" in out.name else ", 이동 제한 10분")
    fig.suptitle(f"충주 3단계: 앞섬 + 짐벌 8초, 통과마다 0.7 × 모델 POD, 계획 폭 190 m, 선회 30° (반경 192 m), {lt}\n"
                 "색 = 5~60분 차량 particle 밀도(탐색 반영 전), 파란 선 = 무인기, 사각형 숫자 = 계획 시각(분), ★ = LKP",
                 fontsize=13)
    fig.savefig(out/"maps.png", dpi=95)
    plt.close(fig)

    st = {s: stats(runs[s]) for s in seeds}
    fig = plt.figure(figsize=(20, 11), layout="constrained")
    gs = fig.add_gridspec(2, 3)
    ax = fig.add_subplot(gs[0, :2])
    grid = np.arange(0, 3601, 60)
    curves = []
    for s in seeds:
        o = runs[s]
        c = np.interp(grid, o["pos_t"], o["pos"], left=0)
        curves.append(c)
        ax.plot(grid/60, c, lw=1, alpha=.6, label=f"{s}")
    ax.plot(grid/60, np.mean(curves, 0), color="black", lw=3, label="평균")
    ax.set_xlabel("LKP 후 시간 (분)"); ax.set_ylabel("누적 POS"); ax.legend(ncol=6, fontsize=8)
    ax.set_title("누적 POS (시나리오별, 검은 선 = 평균)", fontsize=11)
    ax = fig.add_subplot(gs[0, 2])
    x = np.arange(len(seeds))
    ax.bar(x - .2, [runs[s]["pos"][-1] for s in seeds], .4, color="#2a4d7a", label="최종 POS")
    ax.bar(x + .2, [float(runs[s]["escaped"]) for s in seeds], .4, color="#bbb", label="지도 밖으로 나간 비율")
    ax.set_xticks(x); ax.set_xticklabels(seeds); ax.set_xlabel("시나리오"); ax.legend(fontsize=9)
    ax.set_title("최종 POS와 지도 밖 비율", fontsize=11)
    ax = fig.add_subplot(gs[1, :2])
    keys = [("lane", "비행 중 레인 비율"), ("turn_in_lane", "레인 작업 중 선회(롤 10° 이상) 비율"),
            ("fp_in_rect", "레인 작업 중 발자국 중심이 사각형 안인 비율"), ("capped", "12분 상한에 걸린 작업 비율")]
    w = .2
    for k, (key, lab) in enumerate(keys):
        ax.bar(x + (k - 1.5)*w, [st[s][key] for s in seeds], w, label=lab)
    ax.set_xticks(x); ax.set_xticklabels(seeds); ax.set_xlabel("시나리오"); ax.set_ylim(0, 1); ax.legend(fontsize=9, ncol=2)
    ax.set_title("시간을 어디에 썼나", fontsize=11)
    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")
    rows = [["", "평균", "최소", "최대"]]
    for key, lab in [("pos", "최종 POS"), ("esc", "지도 밖")] + [(k, l) for k, l in keys] + [("first", "첫 레인 시작 (분)")]:
        if key == "pos":
            v = [runs[s]["pos"][-1] for s in seeds]
        elif key == "esc":
            v = [float(runs[s]["escaped"]) for s in seeds]
        elif key == "first":
            v = [st[s]["first_lane_min"] for s in seeds]
        else:
            v = [st[s][key] for s in seeds]
        rows.append([lab, f"{np.nanmean(v):.2f}", f"{np.nanmin(v):.2f}", f"{np.nanmax(v):.2f}"])
    tb = ax.table(cellText=rows, loc="center", cellLoc="center", colWidths=[.55, .15, .15, .15])
    tb.auto_set_font_size(False); tb.set_fontsize(10); tb.scale(1, 1.6)
    fig.suptitle(f"충주 3단계 요약 (무인기 1대, 60분, 시나리오 {len(seeds)}개, {out.name})", fontsize=14)
    fig.savefig(out/"summary.png", dpi=95)
    plt.close(fig)
    summary = {int(s): dict(pos=float(runs[s]["pos"][-1]), escaped=float(runs[s]["escaped"]), plans=len(runs[s]["rects"]),
                            pos_15_30_45=[float(np.interp(m*60, runs[s]["pos_t"], runs[s]["pos"])) for m in (15, 30, 45)],
                            **{k: float(v) for k, v in st[s].items()}) for s in seeds}
    (out/"summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    for s, v in summary.items():
        print(s, {k: (round(x, 3) if isinstance(x, float) else x) for k, x in v.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=list(range(1, 11)))
    ap.add_argument("--particles", type=int, default=2000)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--figures-only", action="store_true")
    ap.add_argument("--lane-time", default="flown", choices=["geometry", "flown"])
    ap.add_argument("--max-transit", type=float, default=600.0, help="s; farthest rectangle the planner may pick")
    ap.add_argument("--out", default=None, help="default step3_<lane-time>[_tr<min>]")
    args = ap.parse_args()
    name = args.out or (f"step3_{args.lane_time}" + ("" if args.max_transit == 600 else f"_tr{args.max_transit/60:g}"))
    out = DATA/"chungju"/name
    out.mkdir(parents=True, exist_ok=True)
    if not args.figures_only:
        with get_context("spawn").Pool(args.jobs) as pool:
            pool.starmap(run, [(s, args.particles, args.lane_time, out, args.max_transit) for s in args.seeds])
    figures(args.seeds, args.particles, out)


if __name__ == "__main__":
    main()

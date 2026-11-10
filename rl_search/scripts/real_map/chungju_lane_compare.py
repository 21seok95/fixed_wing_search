"""Step 2: the three lane variants on the SAME rectangle, from the same state (controlled comparison).

The plain SAROPS run is flown until its first plan; that rectangle and the env state at that moment are kept, and
each variant flies the rectangle's lanes from there (no re-planning), so only the lane flying differs:
  plain    UAV on the planned lanes (footprint centre ~510 m ahead of the UAV)
  lead     UAV 510 m behind on each lane, so the footprint centre runs over the planned lane
  prelook  lead + gimbal pointed at the next lane's start from 8 s before each lane end through the turn

Figure in the rectangle frame (lanes horizontal): coverage (open-ground POD), planned lanes, UAV path,
footprint-centre track; along-lane POD profile; summary numbers.

    python -m rl_search.scripts.real_map.chungju_lane_compare --seed 1
"""
from __future__ import annotations

import argparse
import copy
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from rl_search.real_terrain import DATA
from rl_search.scripts.real_map.chungju_sarops import COLORS, VARIANTS

plt.rcParams["font.family"] = ["AppleGothic", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
G = 25.0
PAD = 1300.0


def first_plan(seed, particles, pass_mode="cap"):
    from rl_search.real_scenario import VehicleScenarios, make_env
    from rl_search.sarops import SaropsPlanner
    env = make_env(seed, detection=dict(pass_mode=pass_mode))
    scen = VehicleScenarios(env.sampler).make(seed, particles)
    env.reset(options=dict(scenario=scen, stage=3, t_launch_s=0.0))
    pol = SaropsPlanner(W=133.0, n_starts=3, big_jump=True, search_s=600.0, max_transit_s=600.0, lane_mode="cover",
                        peak_timing="intercept")
    pol.team = [pol]
    while not pol.rects:
        a = float(pol(env))
        if pol.rects:
            break
        env.step([a])
    return env, scen, pol.rects[0][1]


def fly(env, snap, rect, variant, max_s=1500.0):
    from rl_search.accordion import lanes
    env.restore(copy.deepcopy(snap))
    env._nested = 0
    lead, pre, _ = VARIANTS[variant]
    env.lane_lead_m, env.prelook_s = lead, pre
    env.sample_log = []
    pts, fol = lanes(env, rect, 600.0, search_s=600.0)
    s0 = env.surv.copy()
    t0 = env.t
    xy, t_start = [env.plant.pose[:2].copy()], None
    while fol.laps < 1 and env.t < t0 + max_s:
        a = float(fol.action())
        if t_start is None and fol.started:
            t_start = env.t
        env.step([a])
        xy.append(env.plant.pose[:2].copy())
    found = s0 - env.surv
    return dict(pts=pts, uav_pts=getattr(fol, "pts", pts), log=np.array(env.sample_log), xy=np.array(xy),
                t0=t0, t_start=t_start, t_end=env.t, found=found, pos=float(found.sum()/env.belief.n))


def to_frame(rect, x, y):
    c, s = np.cos(rect.theta), np.sin(rect.theta)
    return (x - rect.cx)*c + (y - rect.cy)*s, -(x - rect.cx)*s + (y - rect.cy)*c


def coverage(det, rect, log, t_from):
    from rl_search.detection import PassCap
    from rl_search.scripts.real_map.chungju_zoom import frames
    a = np.arange(-rect.ell/2 - PAD, rect.ell/2 + PAD, G) + G/2
    b = np.arange(-rect.wid/2 - PAD, rect.wid/2 + PAD, G) + G/2
    A, B = np.meshgrid(a, b)
    c, s = np.cos(rect.theta), np.sin(rect.theta)
    px, py = rect.cx + A.ravel()*c - B.ravel()*s, rect.cy + A.ravel()*s + B.ravel()*c
    f, r, u, rate = frames(log)
    miss = np.ones(len(px)); cap = PassCap(len(px), det.cfg)
    for k, (t, x, y, yaw, phi) in enumerate(log[:, :5]):
        if t < t_from:
            continue
        p = det.slice_pod(px, py, x, y, yaw, phi, rate[k], frame=(f[k][None], r[k][None], u[k][None]))
        miss *= 1 - cap.apply(p, det.last_inside)
    return (1 - miss).reshape(A.shape), a, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--particles", type=int, default=2000)
    ap.add_argument("--pass-mode", default="cap", choices=["cap", "scale"])
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    from dataclasses import replace
    from rl_search.detection import DetectionConfig, Detector, load_landcover
    from rl_search.real_scenario import TERRAIN
    from rl_search.terrain import load_terrain

    env, scen, rect = first_plan(args.seed, args.particles, args.pass_mode)
    snap = env.snapshot()
    t_plan = env.t
    print(f"rect {rect.ell:.0f} x {rect.wid:.0f} m at t {env.t/60:.1f} min", flush=True)
    out = {v: fly(env, snap, rect, v) for v in VARIANTS}
    terr = load_terrain(TERRAIN[25.0])
    det = Detector(terr, load_landcover(), replace(DetectionConfig(), vis_shrub=1.0, vis_village=1.0,
                                                    vis_forest_edge=1.0, vis_forest=1.0, oblique_occlusion=False,
                                                    pass_mode=args.pass_mode))

    mode = "통과마다 0.7 × 모델 POD" if args.pass_mode == "scale" else "통과마다 상한 0.7"
    fig = plt.figure(figsize=(22, 34), layout="constrained")
    gs = fig.add_gridspec(5, 3, height_ratios=[1, 1, 1, 1.25, .8])
    rows = {}
    for j, (v, o) in enumerate(out.items()):
        log = o["log"]
        pod, a, b = coverage(det, rect, log, o["t_start"] - 30.0)
        inr = (np.abs(a)[None, :] <= rect.ell/2) & (np.abs(b)[:, None] <= rect.wid/2)
        lane_s = o["t_end"] - o["t_start"]
        rows[v] = dict(pos=o["pos"], pod_in=float(pod[inr].mean()), out_km2=float(pod[~inr].sum()*(G/1000)**2),
                       lane_min=lane_s/60, prof=pod[(np.abs(b)[:, None] <= rect.wid/2)[:, 0]].mean(0), a=a)
        ax = fig.add_subplot(gs[j, :])
        ax.imshow(pod, origin="lower", extent=[a[0] - G/2, a[-1] + G/2, b[0] - G/2, b[-1] + G/2], cmap="magma",
                  vmin=0, vmax=1, interpolation="nearest", aspect="equal")
        ax.add_patch(Rectangle((-rect.ell/2, -rect.wid/2), rect.ell, rect.wid, fill=False, ec="white", lw=2))
        P = o["pts"]
        for i in range(0, len(P) - 1, 2):
            ax.plot(P[i:i + 2, 0], P[i:i + 2, 1], color="#00e5ff", ls="--", lw=1.6,
                    label="계획한 레인 (계획기가 훑는다고 보는 선)" if i == 0 else None)
        m = log[:, 0] >= o["t_start"] - 30.0
        ux, uy = to_frame(rect, o["xy"][:, 0], o["xy"][:, 1])
        ax.plot(ux, uy, color="white", lw=1.0, label="무인기 경로")
        f = log[m, 5:8]
        tc = 500.0/np.maximum(-f[:, 2], 1e-3)
        fx, fy = to_frame(rect, log[m, 1] + tc*f[:, 0], log[m, 2] + tc*f[:, 1])
        ax.plot(fx, fy, ".", color="#ffeb3b", ms=1.5, label="발자국 중심 (1초마다)")
        tg = log[m, 14:16]
        ok = np.isfinite(tg[:, 0])
        if ok.any():
            gx, gy = to_frame(rect, tg[ok, 0], tg[ok, 1])
            ax.plot(gx, gy, "x", color="#00e676", ms=9, mew=2, label="짐벌 지향점")
        ax.set_xlim(a[0], a[-1]); ax.set_ylim(b[0], b[-1])
        r = rows[v]
        ax.set_title(f"{VARIANTS[v][2]}:  사각형 안 평균 POD {r['pod_in']:.2f} · 사각형 밖에 쓴 POD {r['out_km2']:.2f} km² · "
                     f"레인 비행 {r['lane_min']:.1f}분 · 이 작업 POS +{r['pos']:.3f}", fontsize=12, color=COLORS[v])
        ax.set_ylabel("레인 가로 (m)")
        ax.legend(loc="upper left", fontsize=8, ncol=4, facecolor="#333", labelcolor="white")
    from rl_search.scripts.real_map.chungju_zoom import foot
    for j, (v, o) in enumerate(out.items()):
        log = o["log"]
        ax = fig.add_subplot(gs[3, j])
        ax.set_facecolor("#111")
        ax.add_patch(Rectangle((-rect.ell/2, -rect.wid/2), rect.ell, rect.wid, fill=False, ec="white", lw=2))
        P = o["pts"]
        for i in range(0, len(P) - 1, 2):
            ax.plot(P[i:i + 2, 0], P[i:i + 2, 1], color="#00e5ff", ls="--", lw=1.6, label="계획한 레인" if i == 0 else None)
        # one lane change at the left end: the first one that turns there after the lanes start
        m = log[:, 0] >= o["t_start"]
        ux, uy = to_frame(rect, log[:, 1], log[:, 2])
        f = log[:, 5:8]
        tc = 500.0/np.maximum(-f[:, 2], 1e-3)
        fx, fy = to_frame(rect, log[:, 1] + tc*f[:, 0], log[:, 2] + tc*f[:, 1])
        left = np.flatnonzero(m & (ux < -rect.ell/2 + 300))
        k0 = left[0] if len(left) else np.flatnonzero(m)[0]
        seg = np.arange(max(k0 - 40, 0), min(k0 + 110, len(log)))
        ax.plot(ux[m], uy[m], color="white", lw=.6, alpha=.35)
        ax.plot(ux[seg], uy[seg], color="white", lw=2.2, label="무인기 (이 선회 150초)")
        ax.plot(fx[seg], fy[seg], ".", color="#ffeb3b", ms=4, label="발자국 중심")
        for q in seg[::10]:
            Q = np.array(foot(log[q], 500.0))*1000
            qa, qb = to_frame(rect, Q[:, 0], Q[:, 1])
            ax.fill(qa, qb, color=COLORS[v], alpha=.35, ec=COLORS[v], lw=.8)
            ax.plot([ux[q], fx[q]], [uy[q], fy[q]], color="#888", lw=.6)
        tg = log[seg, 14:16]
        ok = np.isfinite(tg[:, 0])
        if ok.any():
            gx, gy = to_frame(rect, tg[ok, 0], tg[ok, 1])
            ax.plot(gx, gy, "x", color="#00e676", ms=12, mew=3, label="짐벌 지향점")
        ax.set_xlim(-rect.ell/2 - 1300, -rect.ell/2 + 900); ax.set_ylim(-rect.wid/2 - 900, rect.wid/2 + 600)
        ax.set_aspect("equal")
        ax.set_title(f"{VARIANTS[v][2]}: 왼쪽 끝 선회 확대\n(색 사각형 = 10초마다 카메라 화면, 회색 선 = 기체→발자국 중심)",
                     fontsize=11, color=COLORS[v])
        ax.set_xlabel("레인 방향 (m)"); ax.set_ylabel("레인 가로 (m)")
        ax.legend(loc="lower right", fontsize=8, facecolor="#333", labelcolor="white")
    ax = fig.add_subplot(gs[4, :2])
    for v, r in rows.items():
        ax.plot(r["a"], r["prof"], color=COLORS[v], lw=2.2, label=VARIANTS[v][2])
    ax.axvspan(-rect.ell/2, rect.ell/2, color="#eee", zorder=0)
    ax.axvline(-rect.ell/2, color="k", lw=.8); ax.axvline(rect.ell/2, color="k", lw=.8)
    ax.set_xlabel("레인 방향 위치 (m, 회색 = 사각형)"); ax.set_ylabel("사각형 폭 안 평균 POD")
    ax.set_title("레인 방향으로 본 POD: 양 끝에서 차이가 난다", fontsize=11); ax.legend()
    ax = fig.add_subplot(gs[4, 2])
    names = list(rows)
    x = np.arange(4)
    for k, v in enumerate(names):
        r = rows[v]
        vals = [r["pod_in"], r["out_km2"]/max(rect.ell*rect.wid/1e6, 1e-9), r["lane_min"]/10, r["pos"]/max(x["pos"] for x in rows.values())]
        ax.bar(x + (k - 1)*.27, vals, .27, color=COLORS[v], label=VARIANTS[v][2])
    ax.set_xticks(x); ax.set_xticklabels(["사각형 안\n평균 POD", "밖에 쓴 POD\n÷ 사각형 넓이", "레인 시간\n÷ 10분", "POS\n÷ 최대"])
    ax.legend(fontsize=8); ax.set_title("요약", fontsize=11)
    fig.suptitle(f"같은 사각형({rect.ell/1000:.1f} × {rect.wid/1000:.2f} km, {t_plan/60:.0f}분 계획), 같은 출발 상태에서 레인 방식만 바꿈: "
                 f"시나리오 {args.seed} (사각형 좌표, 레인을 가로로 돌려 그림, 열린 땅 기준 POD, {mode})", fontsize=13)
    path = DATA/"chungju"/f"lane_compare_{args.seed}_{args.pass_mode}.png"
    fig.savefig(path, dpi=90)
    plt.close(fig)
    for v, r in rows.items():
        print(v, {k: round(float(x), 3) for k, x in r.items() if k not in ("prof", "a")})
    print(f"Saved {path}")


if __name__ == "__main__":
    main()

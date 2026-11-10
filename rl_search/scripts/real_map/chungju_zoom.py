"""Step 2 zoomed comparison: the three lane variants of chungju_sarops.py on one search task, seen from above.

Row 1 (where the area is): weight left on the particles at the middle of the task, as density, with the rectangle
       and the UAV path.
Row 2 (what the camera covered): cumulative single-task POD on open ground (land cover set to open so the footprint
       shape shows), the UAV path, every 20th camera footprint, the rectangle; dots = particles found in the task.
Bottom: POS gained per task, and lane time planned vs flown.

The detection is replayed from the saved 1 s flight log (camera frame per second); the slew rate is taken from the
change of the camera axis between samples, so the replayed POS can differ slightly from the run (printed).

    python -m rl_search.scripts.real_map.chungju_zoom --seed 1 --tasks 0 1
"""
from __future__ import annotations

import argparse
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import LogNorm
from matplotlib.patches import Polygon

from rl_search.real_terrain import DATA
from rl_search.scripts.real_map.chungju_sarops import COLORS, VARIANTS, rect_poly

plt.rcParams["font.family"] = ["AppleGothic", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
G = 25.0          # POD grid (m)


def frames(log):
    f, r, u = log[:, 5:8], log[:, 8:11], log[:, 11:14]
    rate = np.zeros(len(log))
    rate[1:] = np.degrees(np.arccos(np.clip((f[1:]*f[:-1]).sum(1), -1, 1)))/np.maximum(np.diff(log[:, 0]), 1e-6)
    om = np.degrees(9.80665*np.tan(log[:, 4])/33.0)
    return f, r, u, np.maximum(np.abs(om), rate)


def replay(det, belief, log, radius=4000.0):
    """Per-sample particle detection as in RLSearchEnv._detect_oblique. Returns the weight (N,) after each sample
    as a function: w_at(t) and the per-particle found probability between two times."""
    from rl_search.detection import PassCap
    f, r, u, rate = frames(log)
    w = np.ones(belief.n)
    cap = PassCap(belief.n, det.cfg)
    W = np.empty((len(log), belief.n), np.float32)
    D = np.empty((len(log), belief.n), np.float32)      # cumulative found mass per particle
    d = np.zeros(belief.n)
    for k, (t, x, y, yaw, phi) in enumerate(log[:, :5]):
        P = belief.positions(t)
        w[~belief.active(t)] = 0.0
        near = np.flatnonzero((w > 0) & (np.abs(P[:, 0] - x) < radius) & (np.abs(P[:, 1] - y) < radius))
        p = np.zeros(belief.n); inside = np.zeros(belief.n, bool)
        if len(near):
            p[near] = det.slice_pod(P[near, 0], P[near, 1], x, y, yaw, phi, rate[k], frame=(f[k][None], r[k][None], u[k][None]))
            inside[near] = det.last_inside
        q = cap.apply(p, inside)
        d += w*q
        w *= 1 - q
        W[k], D[k] = w, d
    return W, D


def cover(det, log, win):
    """Cumulative POD on a G-metre grid over ``win`` (x0, x1, y0, y1) from the samples in ``log``."""
    from rl_search.detection import PassCap
    xs = np.arange(win[0], win[1], G) + G/2
    ys = np.arange(win[2], win[3], G) + G/2
    GX, GY = np.meshgrid(xs, ys)
    px, py = GX.ravel(), GY.ravel()
    f, r, u, rate = frames(log)
    miss = np.ones(len(px))
    cap = PassCap(len(px), det.cfg)
    for k, (t, x, y, yaw, phi) in enumerate(log[:, :5]):
        p = det.slice_pod(px, py, x, y, yaw, phi, rate[k], frame=(f[k][None], r[k][None], u[k][None]))
        miss *= 1 - cap.apply(p, det.last_inside)
    return (1 - miss).reshape(GX.shape), xs, ys


def foot(row, alt):
    from rl_search.detection import DetectionConfig
    cfg = DetectionConfig()
    th, tv = np.tan(np.radians(cfg.hfov_deg/2)), np.tan(np.radians(cfg.vfov_deg/2))
    x, y = row[1], row[2]
    f, r, u = row[5:8], row[8:11], row[11:14]
    q = []
    for sh, sv in ((-1, -1), (-1, 1), (1, 1), (1, -1)):
        d = f + sh*th*r + sv*tv*u
        t = alt/max(-d[2], 1e-3)
        q.append((x + t*d[0], y + t*d[1]))
    return np.array(q)/1000


def in_rect(px, py, rect):
    cx, cy, ell, wid, th = rect
    c, s = np.cos(th), np.sin(th)
    a = (px - cx)*c + (py - cy)*s
    b = -(px - cx)*s + (py - cy)*c
    return (np.abs(a) <= ell/2) & (np.abs(b) <= wid/2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--tasks", type=int, nargs="*", default=[0, 1])
    ap.add_argument("--half", type=float, default=0.0, help="half window (km); 0 = from the rectangles")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    from dataclasses import replace
    from rl_search.detection import DetectionConfig, Detector, load_landcover
    from rl_search.mission import ParticleBelief
    from rl_search.real_scenario import VehicleScenarios, make_env
    from rl_search.terrain import load_terrain
    from rl_search.real_scenario import TERRAIN

    runs = {v: dict(np.load(DATA/"chungju"/f"sarops_step2_{args.seed}_{v}.npz")) for v in VARIANTS}
    n = int(runs["plain"]["particles"])
    probe = make_env(args.seed)
    scen = VehicleScenarios(probe.sampler).make(args.seed, n)
    terr = load_terrain(TERRAIN[25.0])
    lc = load_landcover()
    det = Detector(terr, lc, DetectionConfig())
    open_cfg = replace(DetectionConfig(), vis_shrub=1.0, vis_village=1.0, vis_forest_edge=1.0, vis_forest=1.0,
                       oblique_occlusion=False)
    det_open = Detector(terr, lc, open_cfg)
    belief = ParticleBelief(scen.track, scen.save_dt_s, scen.exit_step, True)

    for v, o in runs.items():
        o["W"], o["D"] = replay(det, belief, o["log"])
        print(f"{v}: replayed POS {o['D'][-1].sum()/n:.3f} "
              f"(run {o['pos'][-1]:.3f})", flush=True)

    nt = len(args.tasks)
    fig = plt.figure(figsize=(21, 13*nt + 6), layout="constrained")
    gs = fig.add_gridspec(2*nt + 1, 4, width_ratios=[1, 1, 1, .05], height_ratios=[1]*(2*nt) + [.75])
    stats = {v: [] for v in runs}
    for i, k in enumerate(args.tasks):
        spans = []
        for v, o in runs.items():
            if k < len(o["rects"]):
                _, cx, cy, ell, wid, th = o["rects"][k]
                spans.append(max(ell, wid)/2 + 1200)
        half = args.half*1000 or max(spans)
        for j, (v, o) in enumerate(runs.items()):
            if k >= len(o["rects"]):
                continue
            tp, cx, cy, ell, wid, th = o["rects"][k]
            task = o["tasks"][k]
            t_l = task[2]
            t_e = task[3] if np.isfinite(task[3]) else o["log"][-1, 0]
            t_mid = (t_l + t_e)/2
            log = o["log"]
            ks = np.flatnonzero((log[:, 0] >= t_l) & (log[:, 0] <= t_e))
            kt = np.flatnonzero((log[:, 0] >= tp) & (log[:, 0] <= t_e))       # transit + lanes
            win = (cx - half, cx + half, cy - half, cy + half)
            rect = (cx, cy, ell, wid, th)
            # weights at lane start / end
            ka = max(ks[0] - 1, 0) if len(ks) else 0
            kb = ks[-1] if len(ks) else 0
            wa, wb = o["W"][ka], o["W"][kb]
            P = belief.positions(t_mid)
            da, db = o["D"][ka], o["D"][kb]
            gain = float((db - da).sum()/n)
            gain_t = float((db - o["D"][max(kt[0] - 1, 0)]).sum()/n) if len(kt) else 0.0
            mass_in = float(wa[in_rect(P[:, 0], P[:, 1], rect)].sum()/n)
            mass_near = float(wa[(np.abs(P[:, 0] - cx) < half) & (np.abs(P[:, 1] - cy) < half)].sum()/n)
            # row 1: density
            ax = fig.add_subplot(gs[2*i, j])
            bins = [np.arange(win[0], win[1] + 1, 100), np.arange(win[2], win[3] + 1, 100)]
            H, _, _ = np.histogram2d(P[:, 0], P[:, 1], bins=bins, weights=wa/n)
            im1 = ax.imshow(np.where(H > 0, H, np.nan).T, origin="lower", extent=np.array(win)/1000, cmap="YlOrRd",
                            norm=LogNorm(1e-4, 3e-2), interpolation="nearest")
            ax.set_facecolor("#eeeeee")
            xy = o["xy"]/1000; tt = o["tt"]
            m = (tt >= tp) & (tt <= t_e)
            ax.plot(xy[m, 0], xy[m, 1], color=COLORS[v], lw=1.2)
            ax.add_patch(Polygon(rect_poly(*rect), fill=False, ec="black", lw=1.6))
            ax.set_xlim(win[0]/1000, win[1]/1000); ax.set_ylim(win[2]/1000, win[3]/1000); ax.set_aspect("equal")
            ax.set_title(f"{VARIANTS[v][2]} · 작업 {k + 1} ({tp/60:.0f}분 계획, 레인 {t_l/60:.1f}~{t_e/60:.1f}분)\n"
                         f"레인 중간 시각 남은 확률: 사각형 안 {mass_in:.3f}, 창 안 {mass_near:.3f}", fontsize=10)
            # row 2: coverage
            ax = fig.add_subplot(gs[2*i + 1, j])
            pod, xs, ys = cover(det_open, log[ks], win)
            im2 = ax.imshow(pod, origin="lower", extent=np.array(win)/1000, cmap="magma", vmin=0, vmax=1,
                            interpolation="nearest")
            ax.plot(xy[m, 0], xy[m, 1], color="#4fc3f7", lw=1.0)
            ax.add_collection(PolyCollection([foot(log[q], 500.0) for q in ks[::20]], facecolors="none",
                                             edgecolors="white", lw=.5, alpha=.6))
            tgt = log[ks, 14:16]
            pre = np.isfinite(tgt[:, 0])
            if pre.any():
                ax.scatter(tgt[pre, 0]/1000, tgt[pre, 1]/1000, marker="x", s=25, c="#00e676", lw=1.2, zorder=4,
                           label="짐벌 지향점")
                ax.legend(loc="upper right", fontsize=8)
            ax.add_patch(Polygon(rect_poly(*rect), fill=False, ec="#4fc3f7", lw=1.6, ls="--"))
            found = (db - da) > 0.05
            Pe = belief.positions(t_e)
            ax.scatter(Pe[found, 0]/1000, Pe[found, 1]/1000, s=6, c="#00e676", lw=0, zorder=3)
            gx, gy = np.meshgrid(xs, ys)
            inr = in_rect(gx, gy, rect)
            cell = (G/1000)**2
            pod_in = float(pod[inr].mean()) if inr.any() else 0.0
            pod_out = float(pod[~inr].sum()*cell)
            ax.set_xlim(win[0]/1000, win[1]/1000); ax.set_ylim(win[2]/1000, win[3]/1000); ax.set_aspect("equal")
            ax.set_title(f"이 작업 POS +{gain:.3f} (이동 포함 +{gain_t:.3f}), 발견 particle(초록 점) {found.sum()}개\n"
                         f"사각형 안 평균 POD {pod_in:.2f}, 사각형 밖에 쓴 POD {pod_out:.2f} km² (열린 땅 기준)", fontsize=10)
            ax.set_xlabel("동쪽 (km)"); ax.set_ylabel("북쪽 (km)")
            stats[v].append(dict(task=k, gain=gain, gain_t=gain_t, mass_in=mass_in, pod_in=pod_in, pod_out=pod_out,
                                 lane_s=t_e - t_l, area=ell*wid/1e6))
        fig.colorbar(im1, cax=fig.add_subplot(gs[2*i, 3]), label="남은 확률 (100 m 칸)")
        fig.colorbar(im2, cax=fig.add_subplot(gs[2*i + 1, 3]), label="이 작업의 누적 POD (열린 땅)")

    # bottom: POS gained per task, lane time
    ax = fig.add_subplot(gs[-1, :2])
    nk = max(len(o["tasks"]) for o in runs.values())
    wbar = .26
    for j, (v, o) in enumerate(runs.items()):
        g = []
        for k in range(len(o["tasks"])):
            t0 = o["tasks"][k][0]; t1 = o["tasks"][k][3] if np.isfinite(o["tasks"][k][3]) else o["log"][-1, 0]
            ka = np.searchsorted(o["log"][:, 0], t0) - 1; kb = np.searchsorted(o["log"][:, 0], t1, side="right") - 1
            g.append(float((o["D"][kb] - o["D"][max(ka, 0)]).sum()/n))
        ax.bar(np.arange(len(g)) + (j - 1)*wbar, g, wbar, color=COLORS[v], label=f"{VARIANTS[v][2]} (합 {sum(g):.3f})")
    ax.set_xticks(range(nk)); ax.set_xticklabels([f"작업 {k + 1}" for k in range(nk)])
    ax.set_ylabel("작업별 POS (이동 포함)"); ax.legend(fontsize=9); ax.set_title("작업별로 얻은 POS", fontsize=11)
    ax = fig.add_subplot(gs[-1, 2:])
    for j, (v, o) in enumerate(runs.items()):
        lt = [((o["tasks"][k][3] if np.isfinite(o["tasks"][k][3]) else o["log"][-1, 0]) - o["tasks"][k][2])/60
              for k in range(len(o["tasks"]))]
        tr = [(o["tasks"][k][2] - o["tasks"][k][0])/60 for k in range(len(o["tasks"]))]
        x = np.arange(len(lt)) + (j - 1)*wbar
        ax.bar(x, tr, wbar, color=COLORS[v], alpha=.4)
        ax.bar(x, lt, wbar, bottom=tr, color=COLORS[v], label=VARIANTS[v][2])
    ax.axhline(10, color="black", lw=.8, ls=":")
    ax.set_xticks(range(nk)); ax.set_xticklabels([f"작업 {k + 1}" for k in range(nk)])
    ax.set_ylabel("분"); ax.legend(fontsize=9)
    ax.set_title("작업 시간: 옅은 막대 = 이동, 진한 막대 = 레인 (점선 = 계획한 레인 10분)", fontsize=11)
    fig.suptitle(f"충주 2단계 확대 비교: 시나리오 {args.seed}, 기존 레인 / 발자국 앞섬 / 앞섬 + 짐벌 16초 앞당김", fontsize=14)
    path = DATA/"chungju"/f"sarops_zoom_{args.seed}.png"
    fig.savefig(path, dpi=90)
    plt.close(fig)
    for v, s in stats.items():
        for r in s:
            print(v, {a: round(b, 3) if isinstance(b, float) else b for a, b in r.items()})
    print(f"Saved {path}")


if __name__ == "__main__":
    main()

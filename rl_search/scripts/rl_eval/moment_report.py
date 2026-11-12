"""Close-up of chosen minutes of one flight (user 2026-11-12): per interval the path with the camera images (counted for A or not), the A
region (swept / still open) and the particles; the policy's action, roll and reward each 5 s decision; and what the policy saw (POS now and
A value on its 6.4 km map, heading up) at the start and the middle of the interval.

    python -m rl_search.scripts.rl_eval.moment_report --run eval-test-s3-k12-rl-r4_500k_d4 \
        --replay rl_search/data/real/amboy/rl_eval/report/replay_test_k12_r4.npz --minutes 21-26 30-38 53-59
"""
from __future__ import annotations

import argparse
import base64
import gzip
import json
from pathlib import Path

import numpy as np

OUT = Path("rl_search/data/real/amboy/rl_eval/report")
arr = lambda s, dt: np.frombuffer(base64.b64decode(s), dt)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--replay", required=True)
    ap.add_argument("--minutes", nargs="+", required=True, help="intervals after the launch, e.g. 21-26 30-38")
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection, PolyCollection
    from rl_search.scripts.real_map.vehicle_web import RUNS
    plt.rcParams["font.family"] = ["AppleGothic", "sans-serif"]; plt.rcParams["axes.unicode_minus"] = False
    d = json.load(gzip.open(RUNS/args.run/"run.json.gz", "rt"))
    o, res, meta = d["opt"], d["res"], d["meta"]; A = o["area"]
    z = np.load(args.replay)
    tl = float(z["t_launch"]); ts = (z["t"] - tl)/60
    log = z["log"].astype(float); lt = (log[:, 0] - tl)/60
    om_roll = np.degrees(9.80665*np.tan(log[:, 4])/float(z["v"]))
    om_yaw = np.abs(np.degrees(np.gradient(np.unwrap(log[:, 3]), log[:, 0])))
    counted = np.maximum(np.abs(om_roll), om_yaw) <= float(z["rate_max"]) + 1e-9
    quad = z["quad"].astype(float)/1000
    keys, off = arr(A["keys"], "<i4").astype(np.int64), arr(A["off"], "<i4")
    sk, stt = arr(A["sweep_keys"], "<i4").astype(np.int64), arr(A["sweep_t"], "<f4").astype(float)
    first = {}
    for kk, tt in zip(sk, stt):
        first.setdefault(int(kk), tt)
    n, X0, c = A["n"], A["X0"]/1000, A["cell"]/1000
    T, N, dt = res["T"], res["N"], res["dt_s"]
    P = arr(res["pos"], "<i2").reshape(T, N, 2).astype(float)*res["pos_unit_m"]/1000
    Wt = arr(o["weight"], np.uint8).reshape(o["T"], o["N"]).astype(float)/255
    g = np.load("rl_search/data/real/amboy/road_graph_25m.npz")
    pp, oo = g["edge_poly"].astype(float), g["edge_poly_offset"]
    roads = [pp[oo[i]:oo[i + 1]]/1000 for i in range(len(oo) - 1)]
    xy = np.c_[z["x"], z["y"]]/1000
    ivs = [tuple(float(v) for v in s.split("-")) for s in args.minutes]
    fig = plt.figure(figsize=(23, 6.6*len(ivs)), layout="constrained", facecolor="#fcfcfb")
    gs = fig.add_gridspec(len(ivs), 4, width_ratios=[1.35, 1.25, .8, .8])
    for r, (a, b) in enumerate(ivs):
        mid = (a + b)/2
        ax = fig.add_subplot(gs[r, 0]); ax.set_facecolor("#f4f3ef")
        ax.add_collection(LineCollection(roads, colors="#d2d1ca", linewidths=.5, zorder=1))
        tm = tl + 60*mid
        mi = int(np.clip(round((tm - A["min_t0"])/60), 0, A["minutes"] - 1))
        reg = keys[off[mi]:off[mi + 1]]
        sw = np.array([first.get(int(kk), np.inf) <= tm for kk in reg])
        cx_, cy_ = X0 + (reg % n + .5)*c, X0 + (reg//n + .5)*c
        ax.scatter(cx_[~sw], cy_[~sw], s=3, marker="s", color="#57c28d", lw=0, zorder=2)
        ax.scatter(cx_[sw], cy_[sw], s=3, marker="s", color="#c8c8c8", lw=0, zorder=2)
        j = min(int(tm/dt), T - 1, o["T"] - 1)
        ax.scatter(P[j, :, 0], P[j, :, 1], s=2.5, c=Wt[j], cmap="Purples", vmin=-.3, vmax=1, lw=0, zorder=3)
        m1 = (lt >= a) & (lt <= b)
        idx = np.flatnonzero(m1)[::3]
        ax.add_collection(PolyCollection([quad[i] for i in idx if counted[i]], facecolors="none", edgecolors="#1f5f99", linewidths=.5, alpha=.6, zorder=4))
        ax.add_collection(PolyCollection([quad[i] for i in idx if not counted[i]], facecolors="none", edgecolors="#e08a2e", linewidths=.6, alpha=.8, zorder=4))
        ax.plot(xy[:, 0], xy[:, 1], color="#bbbbbb", lw=.6, zorder=5)
        m = (ts >= a) & (ts <= b)
        seg = np.stack([xy[m][:-1], xy[m][1:]], 1)
        lc = LineCollection(seg, cmap="viridis", norm=plt.Normalize(a, b), linewidths=2.2, zorder=6); lc.set_array(ts[m][:-1]); ax.add_collection(lc)
        for mk in np.arange(np.ceil(a), b + .01, 1):
            i = np.searchsorted(ts, mk)
            if i < len(ts):
                ax.annotate(f"{mk:.0f}", xy[i], fontsize=8.5, zorder=7, xytext=(3, 3), textcoords="offset points",
                            bbox=dict(boxstyle="round,pad=.12", fc="white", ec="#999", alpha=.85))
        q = np.r_[xy[m], *[quad[i] for i in idx]] if len(idx) else xy[m]
        cx, cy = (q[:, 0].min() + q[:, 0].max())/2, (q[:, 1].min() + q[:, 1].max())/2; rr = max(np.ptp(q[:, 0]), np.ptp(q[:, 1]))/2 + .4
        ax.set_xlim(cx - rr, cx + rr); ax.set_ylim(cy - rr, cy + rr); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"이륙 후 {a:g}~{b:g}분 · 초록 = {mid:g}분의 A 영역 중 아직 안 센 칸, 회색 = 이미 센 칸\n"
                     f"파랑 테두리 = A에 센 화면, 주황 = 안 센 화면(돌고 있어서) · 점 = particle", loc="left", fontsize=10)
        ax = fig.add_subplot(gs[r, 1])
        ax.step(ts[m], z["a"][m], where="post", color="#0072b2", lw=1.6, label="정책 행동 (−1~1, 5초에 최대 49°)")
        ax.step(ts[m], z["a_applied"][m], where="post", color="#d1495b", lw=1, ls="--", label="실제 적용 (가장자리 마스크 뒤)")
        ax.axhspan(-7.5/49, 7.5/49, color="#1f5f99", alpha=.08)
        ax.plot(ts[m], np.degrees(z["roll"][m])/30, color="#7a5a9e", lw=1, label="롤 / 30°")
        ax.set_ylim(-1.05, 1.05); ax.set_xlim(a, b); ax.grid(alpha=.25); ax.legend(fontsize=8, loc="upper left")
        ax2 = ax.twinx()   # reward in its own units on the right (the step rewards are tiny next to the action)
        ax2.bar(ts[m], 150*z["dPOS"][m], width=5/60*.85, align="edge", color="#0072b2", alpha=.25)
        ax2.bar(ts[m], 100*z["dA"][m], width=5/60*.85, align="edge", bottom=150*z["dPOS"][m], color="#e08a2e", alpha=.45)
        ax2.set_ylabel("5초 보상 (막대: 파랑 POS 몫, 주황 A 몫)", fontsize=8.5)
        ax.set_title("행동(선), 롤(보라), 5초마다 받은 보상(막대) · 옅은 띠 = 1.5°/s 이하가 되는 행동 폭", loc="left", fontsize=10)
        for p, tq in enumerate((a, mid)):
            i = int(np.clip(np.searchsorted(ts, tq), 0, len(ts) - 1))
            ax = fig.add_subplot(gs[r, 2 + p])
            pos_m = z["m6400_pos"][i].astype(float); a_m = z["m6400_A"][i].astype(float)
            rgb = np.ones(pos_m.shape + (3,))
            pn, an = pos_m/max(pos_m.max(), 1e-9), a_m/max(a_m.max(), 1e-9)
            rgb[..., 0] -= .55*pn + .65*an; rgb[..., 1] -= .75*pn + .15*an; rgb[..., 2] -= .25*pn + .65*an
            ax.imshow(np.clip(rgb, 0, 1), extent=(-3.2, 3.2, -3.2, 3.2), origin="upper")
            ax.plot(0, 0, marker="^", ms=10, color="#d1495b", mec="k")
            ax.set_xticks([-3, 0, 3]); ax.set_yticks([-3, 0, 3]); ax.tick_params(labelsize=8)
            ax.set_title(f"정책이 본 6.4 km 지도, {ts[i]:.1f}분\n(위 = 진행 방향, 보라 = 남은 POS, 초록 = A 가치)", loc="left", fontsize=9.5)
    h = meta.get("holdout", {})
    fig.suptitle(f"{meta.get('label', args.run)} [{args.run}] 확대: 구간별 경로·화면·영역, 행동과 보상, 정책이 본 지도", fontsize=13, x=.01, ha="left")
    out = OUT/f"{args.run}_moments.png"
    fig.savefig(out, dpi=78, facecolor="#fcfcfb")
    print("saved", out)
    for a, b in ivs:
        m = (ts >= a) & (ts < b); m1 = (lt >= a) & (lt < b)
        print(f"{a:g}-{b:g} min: decisions {m.sum()}, |a| mean {np.abs(z['a'][m]).mean():.2f}, a sign + {np.mean(z['a'][m] > .05):.0%} / - {np.mean(z['a'][m] < -.05):.0%},"
              f" mask cut {np.abs(z['a'][m] - z['a_applied'][m]).sum():.2f}, |roll| mean {np.degrees(np.abs(z['roll'][m])).mean():.1f} deg,"
              f" images counted {counted[m1].mean():.0%}, reward POS {150*z['dPOS'][m].sum():.2f} A {100*z['dA'][m].sum():.2f}")


if __name__ == "__main__":
    main()

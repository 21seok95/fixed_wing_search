"""Minute-by-minute report of one evaluation flight (user 2026-11-12: look at single flights together to find where the learning falls
short): the path in three 20 min parts over the particles and the A region, the reward the policy got each minute (POS and A parts, on its
own planner particles), the score so far, and the situation each minute (POS so far, region size, distance to the region, turn rate).

    python -m rl_search.scripts.rl_eval.flight_report --run eval-test-s3-k12-rl-r4_500k_d4
Writes rl_search/data/real/amboy/rl_eval/report/<run>.png and .csv (one row per minute after the launch).
"""
from __future__ import annotations

import argparse
import base64
import csv
import gzip
import json
import warnings
from pathlib import Path

import numpy as np

OUT = Path("rl_search/data/real/amboy/rl_eval/report")


def arr(s, dt):
    return np.frombuffer(base64.b64decode(s), dt)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="web run id, e.g. eval-test-s3-k12-rl-r4_500k_d4")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from scipy.spatial import cKDTree
    from rl_search.area_score import RealtimeArea
    from rl_search.real_scenario import VehicleScenarios
    from rl_search.rl_eval_set import REGION, flight_window, make_case, vehicle_config
    from rl_search.rl_real import preload
    from rl_search.scripts.real_map.vehicle_web import RUNS
    from rl_search.scripts.real_map.web_optimize import area_sigma
    plt.rcParams["font.family"] = ["AppleGothic", "sans-serif"]; plt.rcParams["axes.unicode_minus"] = False
    d = json.load(gzip.open(RUNS/args.run/"run.json.gz", "rt"))
    o, res, meta = d["opt"], d["res"], d["meta"]
    s, A, U = o["settings"], o["area"], o["uavs"][0]
    split, k = meta["split"], int(meta["case"])
    st = preload(REGION)
    vs = VehicleScenarios(st["sampler"], vehicle_config(), region=REGION, net=st["net"])
    case, scen, vres, hold = make_case(vs, split, 3, k)
    t_l = float(case.t_launch_s)
    from rl_search.scripts.real_map.web_optimize import realtime_kw
    ra = RealtimeArea(st["terr"], area_sigma(s, 150.0), scen.track, scen.exit_step, scen.save_dt_s, float(s.get("area_x", 70.0)), **realtime_kw(s, REGION),
                      window=flight_window(case.options(scen)))
    # booked A of every sweep (the reward's A part), from the cells the flight swept on its planner particles
    sk, stt = arr(A["sweep_keys"], "<i4").astype(np.int64), arr(A["sweep_t"], "<f4").astype(float)
    ts_u, inv = np.unique(stt, return_inverse=True)
    gain = np.zeros(len(ts_u))
    for i, t in enumerate(ts_u):
        ks = np.unique(sk[inv == i])
        j = ra.frame_now(t)
        Hj = ra.H[ra.j0 + j - 2]
        pos = np.clip(np.searchsorted(Hj, ks), 0, max(len(Hj) - 1, 0))
        inr = Hj[pos] == ks if len(Hj) else np.zeros(len(ks), bool)
        gain[i] = float(ra.F[j - 1][pos[inr]].sum())
    print(f"A replayed {gain.sum():.3f} (stored {A['A']:.3f})")
    lt = arr(U["log_t"], np.float32).astype(float); xyy = arr(U["log_xyyaw"], np.float32).reshape(-1, 3).astype(float)
    om = np.abs(np.degrees(np.gradient(np.unwrap(xyy[:, 2]), lt)))
    pt, pv = np.asarray(o["pos_t"], float), np.asarray(o["pos"], float)
    w_pos = float(s.get("area_w", 1.0))
    mins = np.arange(0, 61)
    tb = t_l + 60*mins
    pos_at = np.interp(tb, pt, pv, left=0.0)
    a_cum = np.array([gain[ts_u < t].sum() for t in tb])
    size_t = (A["j0"] + np.arange(len(A["size_km2"])))*A["dt"]
    keys, off = arr(A["keys"], "<i4").astype(np.int64), arr(A["off"], "<i4")
    n, X0, c = A["n"], A["X0"], A["cell"]
    rows = []
    for m in range(60):
        a, b = tb[m], tb[m + 1]
        sel = (lt >= a) & (lt < b)
        mi = int(np.clip(round((b - A["min_t0"])/60), 0, A["minutes"] - 1))
        reg = keys[off[mi]:off[mi + 1]]
        dist = np.nan
        if sel.any() and len(reg):
            cen = np.stack([X0 + (reg % n + .5)*c, X0 + (reg//n + .5)*c], 1)
            dist = float(cKDTree(cen).query(xyy[sel][-1, :2])[0])
        rows.append(dict(minute=m + 1, battery_left=1 - (m + 1)/60, dPOS=pos_at[m + 1] - pos_at[m], dA=a_cum[m + 1] - a_cum[m],
                         reward=100*((a_cum[m + 1] - a_cum[m]) + w_pos*(pos_at[m + 1] - pos_at[m])), POS=pos_at[m + 1], A=a_cum[m + 1],
                         region_km2=float(np.interp(b, size_t, A["size_km2"])), dist_region_km=dist/1000,
                         turn_dps=float(om[sel].mean()) if sel.any() else np.nan, straight=float(np.mean(om[sel] <= 1.5)) if sel.any() else np.nan))
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT/f"{args.run}.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
    R = {key: np.array([r[key] for r in rows]) for key in rows[0]}
    x = R["minute"] - .5
    first = (ts_u.min() - t_l)/60 if len(ts_u) else 0
    # ---------------------------------------------------------------- figure
    g = np.load("rl_search/data/real/amboy/road_graph_25m.npz")
    pp, oo = g["edge_poly"].astype(float), g["edge_poly_offset"]
    roads = [pp[oo[i]:oo[i + 1]]/1000 for i in range(len(oo) - 1)]
    T, N, dt = res["T"], res["N"], res["dt_s"]
    P = arr(res["pos"], "<i2").reshape(T, N, 2).astype(float)*res["pos_unit_m"]/1000
    W = arr(o["weight"], np.uint8).reshape(o["T"], o["N"]).astype(float)/255
    fig = plt.figure(figsize=(21, 15.5), layout="constrained", facecolor="#fcfcfb")
    gs = fig.add_gridspec(3, 3, height_ratios=[1.35, 1, 1])
    fly = lt >= t_l
    xy = xyy[fly, :2]/1000; tt = (lt[fly] - t_l)/60
    span = np.r_[xy.min(0), xy.max(0)]
    for p, (a, b) in enumerate(((0, 20), (20, 40), (40, 60))):
        ax = fig.add_subplot(gs[0, p]); ax.set_facecolor("#f4f3ef")
        ax.add_collection(LineCollection(roads, colors="#d2d1ca", linewidths=.5, zorder=1))
        tm = t_l + 60*b
        j = min(int(tm/dt), T - 1, o["T"] - 1)
        mi = int(np.clip(round((tm - A["min_t0"])/60), 0, A["minutes"] - 1))
        reg = keys[off[mi]:off[mi + 1]]
        ax.scatter(X0/1000 + (reg % n + .5)*c/1000, X0/1000 + (reg//n + .5)*c/1000, s=1.5, marker="s", color="#bfe3cf", lw=0, zorder=2)
        ax.scatter(P[j, :, 0], P[j, :, 1], s=2, c=W[j], cmap="Purples", vmin=-.3, vmax=1, lw=0, zorder=3)
        ax.plot(xy[:, 0], xy[:, 1], color="#c9c9c9", lw=.6, zorder=4)
        m = (tt >= a) & (tt <= b)
        seg = np.stack([xy[m][:-1], xy[m][1:]], 1)
        lc = LineCollection(seg, cmap="viridis", norm=plt.Normalize(a, b), linewidths=1.8, zorder=5); lc.set_array(tt[m][:-1]); ax.add_collection(lc)
        for mk in range(a, b + 1, 5):
            i = np.searchsorted(tt, mk)
            if i < len(tt):
                ax.annotate(str(mk), xy[i], fontsize=8.5, zorder=6, xytext=(3, 3), textcoords="offset points",
                            bbox=dict(boxstyle="round,pad=.12", fc="white", ec="#999", alpha=.85))
        ax.plot(*np.asarray(case.lkp)/1000, marker="*", ms=12, color="#ffd400", mec="k", zorder=7)
        q = xy[m] if m.any() else xy
        cx, cy = (q[:, 0].min() + q[:, 0].max())/2, (q[:, 1].min() + q[:, 1].max())/2
        r = max(np.ptp(q[:, 0]), np.ptp(q[:, 1]))/2 + 1.0
        ax.set_xlim(cx - r, cx + r); ax.set_ylim(cy - r, cy + r); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.plot([cx - r + .3*r/4, cx - r + .3*r/4 + 1], [cy - r + .15*r/4]*2, color="k", lw=3, zorder=8)
        ax.text(cx - r + .3*r/4, cy - r + .3*r/4, "1 km", fontsize=8.5, zorder=8)
        ax.set_title(f"이륙 후 {a}~{b}분 (숫자 = 이륙 후 분, 회색 = 다른 시간의 경로)", loc="left", fontsize=10.5)
    def shade(ax):
        ax.axvspan(0, first, color="#ddd", alpha=.5, lw=0); ax.set_xlim(0, 60); ax.grid(alpha=.25)
    ax = fig.add_subplot(gs[1, :2])
    ax.bar(x, 100*w_pos*R["dPOS"], width=.85, color="#0072b2", label=f"POS 몫 (100 × {w_pos:g} × ΔPOS)")
    ax.bar(x, 100*R["dA"], width=.85, bottom=100*w_pos*R["dPOS"], color="#e08a2e", label="A 몫 (100 × ΔA)")
    shade(ax); ax.set_ylabel("1분 동안 받은 보상"); ax.legend(fontsize=9)
    ax.set_title(f"(가) 분마다 받은 보상 (계획 particle 기준) · 회색 = 처음 칸을 셀 때까지({first:.0f}분)", loc="left", fontsize=11)
    ax = fig.add_subplot(gs[1, 2])
    ax.plot(R["minute"], w_pos*R["POS"], color="#0072b2", lw=2, label=f"{w_pos:g} × POS")
    ax.plot(R["minute"], R["A"], color="#e08a2e", lw=2, label="A"); ax.plot(R["minute"], R["A"] + w_pos*R["POS"], color="#111", lw=2.4, label="총점")
    shade(ax); ax.legend(fontsize=9); ax.set_title("(나) 지금까지의 점수", loc="left", fontsize=11)
    for p, (key, lab, unit) in enumerate((("region_km2", "(다) 그 순간 A 영역 넓이", "km²"), ("dist_region_km", "(라) 기체에서 A 영역까지 거리", "km"),
                                          ("turn_dps", "(마) 평균 선회율과 곧게 난 비율", "°/s"))):
        ax = fig.add_subplot(gs[2, p])
        ax.plot(R["minute"] - .5, R[key], color="#5b3f8c", lw=2)
        if key == "turn_dps":
            ax.axhline(1.5, color="#e08a2e", ls="--", lw=1)
            ax2 = ax.inset_axes([0, -0.38, 1, .25]); ax2.bar(x, 100*R["straight"], color="#1f5f99", width=.85); ax2.set_xlim(0, 60); ax2.set_ylim(0, 100)
            ax2.set_ylabel("≤1.5°/s %", fontsize=8); ax2.tick_params(labelsize=8)
        shade(ax); ax.set_ylabel(unit); ax.set_title(lab, loc="left", fontsize=11)
        ax.set_xlabel("이륙 후 분 (배터리 = 남은 분 / 60)" if key != "turn_dps" else "")
    h = meta.get("holdout", {})
    fig.suptitle(f"{meta.get('label', args.run)} [{args.run}]\n지도: 점 = 그 구간 끝의 particle(진할수록 남은 확률 큼), 연두 = 그 구간 끝의 A 영역, ★ = LKP · hold-out POS {h.get('pos', float('nan')):.3f} · A {h.get('A', float('nan')):.3f} · 총점 {h.get('total', float('nan')):.3f}"
                 f" · 출발 {case.info.get('launch_dist_m', 0)/1000:.0f} km, 출동 지연 {t_l/60:.0f}분", fontsize=13, x=.01, ha="left")
    fig.savefig(OUT/f"{args.run}.png", dpi=80, facecolor="#fcfcfb")
    print("saved", OUT/f"{args.run}.png")
    for a in range(0, 60, 10):
        sl = slice(a, a + 10)
        print(f"{a:2d}-{a + 10:2d} min: reward {R['reward'][sl].sum():6.1f} (POS part {100*w_pos*R['dPOS'][sl].sum():5.1f}, A part {100*R['dA'][sl].sum():5.1f}),"
              f" POS at end {R['POS'][sl][-1]:.3f}, region {np.nanmean(R['region_km2'][sl]):.1f} km2, dist {np.nanmean(R['dist_region_km'][sl]):.2f} km,"
              f" turn {np.nanmean(R['turn_dps'][sl]):.1f} deg/s, straight {np.nanmean(R['straight'][sl]):.0%}")


if __name__ == "__main__":
    main()

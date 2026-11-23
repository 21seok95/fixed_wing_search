"""Do the observation maps' leads match when the UAV actually reaches what a map shows? (user 2026-11-19, before deciding on several
leads per map or an intercept map; docs/scenario_redesign.md "관측 지도의 lead 확인").

From saved RL flights (web runs: the particles every 10 s, the planner-particle survival every 10 s, the UAV track): every detection
(a particle's survival dropping between two frames, weighted by the probability found) is looked back on from decision times 1, 2, 4,
... min before it. At each look-back the particle's distance from the UAV says on which map it first appears (the smallest map whose
half side holds it); the look-back time is how far ahead that map would have had to show it. Compared with the map's lead and with the
straight-flight time distance / 33 m/s.

    python -m rl_search.scripts.rl_eval.lead_check --runs eval-v2j-s3-k0-rl-r15_chungju-4000000 eval-v2j-s3-k1-rl-r15_chungju-4000000 ...
Writes report/lead_check.png and .json.
"""
from __future__ import annotations

import argparse
import base64
import gzip
import json
from pathlib import Path

import numpy as np

OUT = Path("rl_search/data/real/amboy/rl_eval/report")
MAPS7 = (("1.6 km", 1600.0, 15.0), ("3.2 km", 3200.0, 30.0), ("6.4 km", 6400.0, 60.0), ("12.8 km", 12800.0, 150.0),
         ("25.6 km", 25600.0, 300.0), ("51.2 km", 51200.0, 600.0), ("102.4 km", 102400.0, 600.0))
V_UAV = 33.3


def events(run_dir: Path):
    """(time of detection s, found mass, particle index, frame) per survival drop, the particle positions (T, N, 2) m, the UAV track (t, xy)."""
    r = json.load(gzip.open(run_dir/"run.json.gz", "rt"))
    o, res = r["opt"], r["res"]
    T, N = o["T"], o["N"]
    W = np.frombuffer(base64.b64decode(o["weight"]), np.uint8).reshape(T, N).astype(float)/255.0
    P = np.frombuffer(base64.b64decode(res["pos"]), "<i2").reshape(res["T"], N, 2).astype(float)*res["pos_unit_m"]
    u = o["uavs"][0]
    t = np.frombuffer(base64.b64decode(u["log_t"]), np.float32).astype(float)
    xy = np.frombuffer(base64.b64decode(u["log_xyyaw"]), np.float32).reshape(-1, 3)[:, :2].astype(float)
    drop = np.clip(W[:-1] - W[1:], 0, None)                    # found between frame k and k + 1
    k, i = np.nonzero(drop > 1e-4)
    return dict(dt=o["dt_s"], td=(k + 1)*o["dt_s"], m=drop[k, i]/N, i=i, k=k + 1, P=P, t=t, xy=xy, pos=r["meta"]["holdout"]["pos"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--name", default="R15 400만, 지리산 v2j 3개")
    a = ap.parse_args()
    from rl_search.scripts.real_map.vehicle_web import RUNS
    lags = np.array([60, 120, 240, 480, 960, 1920])            # look-back times (s)
    rows = []                                                  # (lag s, distance m, mass, first map index)
    halves = np.array([s/2 for _, s, _ in MAPS7])
    for rid in a.runs:
        e = events(RUNS/rid)
        for lag in lags:
            tl = e["td"] - lag
            ok = tl >= e["t"][0]
            fr = np.clip(np.round(tl/e["dt"]).astype(int), 0, e["P"].shape[0] - 1)
            ux = np.interp(tl, e["t"], e["xy"][:, 0]); uy = np.interp(tl, e["t"], e["xy"][:, 1])
            p = e["P"][fr, e["i"]]
            d = np.hypot(p[:, 0] - ux, p[:, 1] - uy)
            mi = np.searchsorted(halves, d)                    # the smallest map whose half side holds it (7 = none)
            rows += list(zip(np.full(ok.sum(), lag), d[ok], e["m"][ok], mi[ok]))
    R = np.array(rows)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["AppleGothic", "sans-serif"]; plt.rcParams["axes.unicode_minus"] = False
    INK, MUTED, GRID, BG = "#1f1f1e", "#6b6b67", "#e4e3de", "#fcfcfb"
    fig, ax = plt.subplots(1, 2, figsize=(17, 6.6), facecolor=BG)
    # left: for every look-back, the distance the detected mass was at (weighted median and 25-75 %)
    med = []
    for lag in lags:
        s = R[R[:, 0] == lag]
        o = np.argsort(s[:, 1]); c = np.cumsum(s[o, 2])/s[:, 2].sum()
        q = [s[o, 1][np.searchsorted(c, x)] for x in (.25, .5, .75)]
        med.append(q)
    med = np.array(med)/1000
    a0 = ax[0]
    a0.fill_between(lags/60, med[:, 0], med[:, 2], color="#2a78d6", alpha=.2, label="실제: 찾기 전 그 시점의 거리 (25~75%, 찾은 확률 가중)")
    a0.plot(lags/60, med[:, 1], "o-", color="#2a78d6", lw=2, label="실제: 중앙값")
    tt = np.linspace(0, 32, 50)
    a0.plot(tt, V_UAV*tt*60/1000, color=MUTED, ls="--", label="곧장 날아갈 때 (거리 = 33 m/s × 시간)")
    for n, s, lead in MAPS7:
        a0.plot(lead/60, s/2/1000, "s", color="#d6336c", ms=8)
        a0.text(lead/60 + .3, s/2/1000, n, fontsize=9, color="#d6336c", va="center")
    a0.plot([], [], "s", color="#d6336c", label="지도의 lead (가로)와 반 변 (세로)")
    a0.set_xscale("symlog", linthresh=1); a0.set_yscale("log")
    a0.set_xlabel("찾기 몇 분 전에 봤나 (= 지도가 보여 줘야 할 미래, 분)", color=MUTED); a0.set_ylabel("그때 기체에서 거리 (km)", color=MUTED)
    a0.set_title("찾은 입자는 찾기 몇 분 전에 얼마나 멀리 있었나", loc="left", fontsize=12, color=INK)
    a0.legend(frameon=False, fontsize=9, loc="lower right")
    # right: per map, the look-back time of the mass that first appears on that map (distance between the smaller map's half and its own)
    a1 = ax[1]
    names = [n for n, _, _ in MAPS7]
    for k, (n, s, lead) in enumerate(MAPS7):
        sel = R[:, 3] == k
        if not sel.any():
            continue
        w = R[sel, 2]; L = R[sel, 0]
        share = np.array([w[L == lag].sum() for lag in lags])/w.sum()
        a1.scatter(np.full(len(lags), k), lags/60, s=2000*share, color="#2a78d6", alpha=.6, lw=0)
        mean = float((L*w).sum()/w.sum())/60
        a1.plot(k, mean, "_", color="#2a78d6", ms=26, mew=2.5)
        a1.plot(k, lead/60, "s", color="#d6336c", ms=9)
        a1.plot(k, s/2/V_UAV/60, "^", color=MUTED, ms=9)
    a1.plot([], [], "o", color="#2a78d6", alpha=.6, label="실제: 그 지도에 처음 들어온 입자를 찾기까지 (원 크기 = 찾은 확률 몫)")
    a1.plot([], [], "_", color="#2a78d6", ms=20, mew=2.5, label="실제 평균")
    a1.plot([], [], "s", color="#d6336c", label="지금 lead")
    a1.plot([], [], "^", color=MUTED, label="지도 끝까지 곧장 가는 시간")
    a1.set_xticks(range(len(names)), names); a1.set_yscale("log"); a1.set_ylabel("분", color=MUTED)
    a1.set_title("지도별: 지금 lead 대 실제로 찾기까지 걸린 시간", loc="left", fontsize=12, color=INK)
    a1.legend(frameon=False, fontsize=9, loc="upper left")
    for x in ax:
        x.set_facecolor(BG); x.grid(color=GRID, lw=.5); x.tick_params(colors=MUTED)
        for sp in x.spines.values():
            sp.set_color(GRID)
    fig.suptitle(f"관측 지도의 lead 확인 ({a.name}): 찾은 입자를 찾기 1~32분 전으로 되돌려 봄", x=.01, ha="left", fontsize=13.5, color=INK)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT/"lead_check.png", dpi=80, facecolor=BG)
    (OUT/"lead_check.json").write_text(json.dumps(dict(runs=a.runs, lags=lags.tolist(), dist_q_km=med.round(3).tolist())))
    print("lag (min) -> distance km 25/50/75%:", {int(l/60): tuple(np.round(m, 1)) for l, m in zip(lags, med)})


if __name__ == "__main__":
    main()

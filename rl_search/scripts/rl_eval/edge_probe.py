"""The map edge as open ground (user 2026-11-17): the user's Chungju web flights of R14 road flown again as the web viewer flies them, with
  B - particles that leave the map going on straight (VehicleConfig.exit_straight; the belief keeps them)
  C - B, and the ground off the map seen as open (visibility 1) both by the policy's observation and by the detection (setting
      ``off_map_vis``; before: 0 in the observation, the nearest edge cell's visibility in the detection)
Saved as web runs next to the originals; figure report/edge_probe.png (paths of original / B / C, time near the edge and off the map,
circling, loitering = 10 min net displacement / distance flown, POS).

    python -m rl_search.scripts.rl_eval.edge_probe run --variant B      (and C)
    python -m rl_search.scripts.rl_eval.edge_probe figure
"""
from __future__ import annotations

import argparse
import io
import json
import time
import urllib.request
import warnings

import numpy as np

TAG = {"B": "시험: 지도 밖 직진", "C": "시험: 지도 밖 직진 + 밖 가시도 1", "D": "시험 D: 직진 + 안팎 가시도 1"}


class _Q:
    def put(self, m):
        pass


def run(variant: str):
    warnings.filterwarnings("ignore")
    import rl_search.scripts.rl_eval.signal_check as sc
    from rl_search.scripts.real_map import web_rl
    from rl_search.scripts.real_map.vehicle_web import run_meta, save_run
    from rl_search.scripts.rl_eval.circling_cause import web_runs
    base = sc.settings_of
    if variant in ("C", "D"):
        sc.settings_of = lambda cfg: base(cfg) | dict(off_map_vis=1.0)
    if variant == "D":                              # the policy also sees the ground inside the map as open (detection keeps the real one)
        from rl_search.rl_real import RealRLEnv
        orig = RealRLEnv.observation

        def observation(self):
            o = orig(self)
            for k in o:
                if k != "vec":
                    o[k][4] = 1.0
            return o
        RealRLEnv.observation = observation
    stamp = time.strftime("%Y%m%d-%H%M%S")
    for i, m in enumerate(web_runs()):
        if "시험" in m.get("label", ""):
            continue
        cfg = dict(m["config"], exit_straight=True)
        p = dict(view=m["view"], config=cfg, model=m["model"], launch_min=m.get("launch_min", 0.0), kind="rl")
        t0 = time.perf_counter()
        data = web_rl._run(p, _Q())
        data["opt"]["sec"] = round(time.perf_counter() - t0, 1)
        extra = data.pop("meta_extra", {})
        rid = f"{stamp}-{dict(B=800, C=700, D=600)[variant] + i:03d}"
        label = f"{m['label']} · {TAG[variant]} (원래 {m['id'][9:15]})"
        meta = run_meta(data["opt"], rid, label, time.strftime("%m-%d %H:%M"), m["view"], cfg, group="web", **extra)
        save_run(rid, data, meta)
        print(f"{variant} {i + 1}: {rid} POS {data['opt']['final']:.3f} (web {m['final']:.3f})", flush=True)


def metrics(xy, tt, half=30000.0):
    xy, tt = np.asarray(xy, float), np.asarray(tt, float)
    step = np.hypot(*np.diff(xy, axis=0).T)
    cum = np.concatenate([[0.0], np.cumsum(step)])
    ratios = []
    for j in range(len(xy)):                       # every point that has 10 min of flight after it
        e = np.searchsorted(tt, tt[j] + 600.0)
        if e >= len(xy):
            break
        ratios.append(np.hypot(*(xy[e] - xy[j]))/max(cum[e] - cum[j], 1.0))
    return dict(edge=float((np.abs(xy).max(axis=1) > half - 2000).mean()), off=float((np.abs(xy).max(axis=1) > half).mean()),
                loiter=float(np.mean(np.array(ratios) < 0.2)) if ratios else 0.0)


def figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from rl_search.scripts.rl_eval.chungju_web_runs import load
    plt.rcParams["font.family"] = ["AppleGothic", "NanumGothic", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False
    rows = load("20261117")
    orig = [r for r in rows if "시험" not in r["label"]]
    by = {v: {r["label"].split("(원래 ")[1][:6]: r for r in rows if TAG[v] + " (" in r["label"]} for v in ("B", "C", "D")}
    info = json.loads(urllib.request.urlopen("http://127.0.0.1:8000/api/region?key=chungju", timeout=60).read())
    img = plt.imread(io.BytesIO(urllib.request.urlopen("http://127.0.0.1:8000/map.png?region=chungju", timeout=120).read()))
    ext = info["extent"]
    n = len(orig)
    fig, axs = plt.subplots(n, 4, figsize=(20, 5*n), facecolor="#fcfcfb")
    table = []
    for i, r in enumerate(orig):
        runs = [("원래 (웹)", r), ("지도 밖 직진", by["B"].get(r["id"][9:15])), ("직진 + 밖 가시도 1", by["C"].get(r["id"][9:15])),
                ("D: 직진 + 안팎 가시도 1", by["D"].get(r["id"][9:15]))]
        pts = np.vstack([x["xy"]/1000 for _, x in runs if x is not None] + [r["P30"]/1000])
        lo, hi = np.percentile(pts, 1, axis=0), np.percentile(pts, 99, axis=0)
        c, h = (lo + hi)/2, max((hi - lo).max()/2 + 2, 6)
        row = [i + 1]
        for j, (nm, x) in enumerate(runs):
            ax = axs[i, j]
            if x is None:
                ax.axis("off"); row += [np.nan]*4; continue
            xy = x["xy"]/1000
            ax.imshow(img, extent=ext, origin="upper", alpha=0.8)
            w = x["w_end"]
            ax.scatter(x["P_end"][:, 0]/1000, x["P_end"][:, 1]/1000, s=1 + 10*w, c=w, cmap="autumn_r", vmin=0, vmax=1, lw=0, alpha=0.6)
            m3 = len(xy)//3
            for a, b, col in ((0, m3, "#1f3a93"), (m3, 2*m3, "#7b2cbf"), (2*m3, len(xy), "#000000")):
                ax.plot(xy[a:b + 1, 0], xy[a:b + 1, 1], color=col, lw=1.0)
            ax.plot([-30, 30, 30, -30, -30], [-30, -30, 30, 30, -30], color="#d6336c", lw=0.8, ls="--")
            ax.plot(*(x["lkp"]/1000), marker="*", ms=12, color="#d6336c", mec="white")
            ax.set_xlim(c[0] - h, c[0] + h); ax.set_ylim(c[1] - h, c[1] + h); ax.set_aspect("equal")
            mt = metrics(x["xy"], x["tt"])
            ax.set_title(f"{i + 1}번 {nm}: POS {x['final']:.2f}\n가장자리 {100*mt['edge']:.0f}%, 지도 밖 {100*mt['off']:.0f}%, 맴돎 {100*mt['loiter']:.0f}%",
                         loc="left", fontsize=10)
            ax.tick_params(labelsize=7)
            row += [x["final"], mt["edge"], mt["off"], mt["loiter"]]
        table.append(row)
    fig.suptitle("충주 15개: 원래 / 지도 밖 직진 / 직진 + 지도 밖 가시도 1 / D: 직진 + 안팎 가시도 1 (빨간 점선 = 지도 경계; 맴돎 = 10분 동안 실제 이동이 날아간 거리의 20% 미만인 구간)",
                 x=0.01, ha="left", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.995))
    fig.savefig("rl_search/data/real/amboy/rl_eval/report/edge_probe.png", dpi=55, facecolor="#fcfcfb")
    T = np.array(table, float)
    names = ("원래", "직진", "직진+밖1", "D")
    for j, nm in enumerate(names):
        c0 = 1 + 4*j
        print(f"{nm}: POS {np.nanmean(T[:, c0]):.3f}, edge {100*np.nanmean(T[:, c0 + 1]):.1f}%, off-map {100*np.nanmean(T[:, c0 + 2]):.1f}%, "
              f"loiter {100*np.nanmean(T[:, c0 + 3]):.1f}%")
    for row in T:
        print(int(row[0]), " | ".join(f"{row[1 + 4*j]:.2f} e{100*row[2 + 4*j]:.0f} o{100*row[3 + 4*j]:.0f} l{100*row[4 + 4*j]:.0f}" for j in range(4)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=("run", "figure"))
    ap.add_argument("--variant", choices=("B", "C", "D"), default="B")
    a = ap.parse_args()
    run(a.variant) if a.what == "run" else figure()


if __name__ == "__main__":
    main()

"""The RL flights the user ran in the web viewer on the Chungju map (generalisation check, 2026-11-17): every stored run of that day on
chungju, its UAV path over the map with the particles (remaining weight at the end of the flight), the POS over time, and the target
settings that differ between runs.

    python -m rl_search.scripts.rl_eval.chungju_web_runs [--day 20261117]
"""
from __future__ import annotations

import argparse
import base64
import gzip
import io
import json
import urllib.request
from pathlib import Path

import numpy as np

RUNS = Path("rl_search/data/real/chungju/web_runs")
OUT = Path("rl_search/data/real/amboy/rl_eval/report")


def dec(s, dt):
    return np.frombuffer(base64.b64decode(s), dtype=dt)


def load(day: str):
    rows = []
    for d in sorted(RUNS.glob(f"{day}-*")):
        m = json.loads((d/"meta.json").read_text())
        if (m.get("view") or {}).get("region") != "chungju" or "RL" not in m.get("method", m.get("label", "")):
            continue
        r = json.loads(gzip.decompress((d/"run.json.gz").read_bytes()))
        o, res = r["opt"], r["res"]
        T, N = res["T"], res["N"]
        P = dec(res["pos"], "<i2").reshape(T, N, 2).astype(float)*res["pos_unit_m"]
        To = o["T"]
        W = dec(o["weight"], np.uint8).reshape(To, N).astype(float)/255
        u = o["uavs"][0]
        xy = dec(u["xy"], np.float32).reshape(-1, 2)
        tt = dec(u["tt"], np.float32).astype(float)
        st = dec(res["state"], np.uint8).reshape(T, N)
        k = min(To, T) - 1
        rows.append(dict(id=m["id"], label=m["label"], method=m.get("method", ""), view=m["view"], config=m["config"], final=float(o["final"]),
                         holdout=(m.get("holdout") or {}).get("pos"), pos_t=np.array(o["pos_t"]), pos=np.array(o["pos"]), xy=xy, tt=tt,
                         P_end=P[k], w_end=W[k], P30=P[min(180, T - 1)], start=np.array(o["start"]), lkp=np.array(o["lkp"]),
                         off_share=float((st[k] == 2).mean()) if st.size else 0.0, escaped=float(o.get("escaped", 0.0))))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--day", default="20261117")
    a = ap.parse_args()
    rows = load(a.day)
    info = json.loads(urllib.request.urlopen("http://127.0.0.1:8000/api/region?key=chungju", timeout=60).read())
    png = urllib.request.urlopen("http://127.0.0.1:8000/map.png?region=chungju", timeout=120).read()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["AppleGothic", "NanumGothic", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False
    INK, MUTED, GRID, BG = "#1f1f1e", "#6b6b67", "#e4e3de", "#fcfcfb"
    img = plt.imread(io.BytesIO(png))
    ext = info["extent"]                                       # km: x0, x1, y0, y1
    # target settings that differ between the runs
    keys = sorted({k for r in rows for k in r["config"]})
    diff = [k for k in keys if len({json.dumps(r["config"].get(k)) for r in rows}) > 1]
    n = len(rows)
    nc = 5
    nr = int(np.ceil(n/nc))
    fig, axs = plt.subplots(nr + 1, nc, figsize=(5.0*nc, 5.0*(nr + 1)), facecolor=BG, squeeze=False,
                            gridspec_kw=dict(height_ratios=[1]*nr + [0.9]))
    for ax, (i, r) in zip(axs[:nr].ravel(), enumerate(rows)):
        xy = r["xy"]/1000
        pts = np.vstack([xy, r["P30"]/1000, [r["start"]/1000, r["lkp"]/1000]])
        lo, hi = np.percentile(pts, 1, axis=0), np.percentile(pts, 99, axis=0)
        c, h = (lo + hi)/2, max((hi - lo).max()/2 + 1.5, 4)
        ax.imshow(img, extent=ext, origin="upper", alpha=0.85)
        w = r["w_end"]
        ax.scatter(r["P_end"][:, 0]/1000, r["P_end"][:, 1]/1000, s=2 + 14*w, c=w, cmap="autumn_r", vmin=0, vmax=1, lw=0, alpha=0.7, zorder=2)
        ax.plot(xy[:, 0], xy[:, 1], color="#1f3a93", lw=1.1, zorder=3)
        ax.plot(*(r["start"]/1000), marker="^", ms=9, color=INK, mec="white", zorder=4)
        ax.plot(*(r["lkp"]/1000), marker="*", ms=13, color="#d6336c", mec="white", zorder=4)
        ax.set_xlim(c[0] - h, c[0] + h); ax.set_ylim(c[1] - h, c[1] + h); ax.set_aspect("equal")
        dl = float(np.hypot(*(r["start"] - r["lkp"])))/1000
        ax.set_title(f"{i + 1}번  POS {r['final']:.2f}   이륙 → LKP {dl:.0f} km\nLKP ({r['lkp'][0]/1000:.1f}, {r['lkp'][1]/1000:.1f}) km, 도주 방향 {r['view']['heading']:.0f}°, "
                     f"방향 퍼짐 {r['config'].get('heading_std_deg')}°, 우회 {r['config'].get('p_detour')}", loc="left", fontsize=9.5, color=INK)
        ax.tick_params(colors=MUTED, labelsize=7)
    for ax in axs[:nr].ravel()[n:]:
        ax.axis("off")
    ax = axs[nr, 0]
    for i, r in enumerate(rows):
        ax.plot(r["pos_t"]/60, r["pos"], lw=1.2, label=f"{i + 1}")
    ax.set_xlabel("신고 뒤 분", fontsize=8.5, color=MUTED); ax.set_ylabel("찾은 POS", fontsize=8.5, color=MUTED)
    ax.set_title("POS (번호 = 위 그림)", loc="left", fontsize=10, color=INK)
    ax.legend(fontsize=7, ncol=3, frameon=False)
    axs[nr, 1].axis("off")
    txt = ["비행마다 다른 표적 설정:"] + [f"  {k}: " + ", ".join(str(r['config'].get(k)) for r in rows) for k in diff]
    axs[nr, 1].axis("off")
    for ax in axs[nr, 2:]:
        ax.axis("off")
    fig.suptitle(f"웹에서 충주에 날린 RL 비행 {n}개 ({rows[0]['method'] if rows else ''}): 파란 선 = 기체, 점 = 비행 끝에 남은 입자(노랑→빨강 = 많이 남음), "
                 f"▲ 이륙, ★ LKP", x=0.01, ha="left", fontsize=13, color=INK)
    fig.subplots_adjust(left=0.03, right=0.99, top=0.93, bottom=0.04, wspace=0.18, hspace=0.32)
    fig.savefig(OUT/"chungju_web_runs.png", dpi=75, facecolor=BG)
    for i, r in enumerate(rows):
        print(i + 1, r["id"], f"POS {r['final']:.3f}", "LKP", np.round(r["lkp"]/1000, 1), "heading", r["view"]["heading"], "start", np.round(r["start"]/1000, 1),
              f"off-road {100*r['off_share']:.0f}%", f"escaped {r['escaped']:.3f}", "path km", round(float(np.hypot(*np.diff(r["xy"], axis=0).T).sum()/1000), 1))
    print("differing settings:", {k: [r["config"].get(k) for r in rows] for k in diff})


if __name__ == "__main__":
    main()

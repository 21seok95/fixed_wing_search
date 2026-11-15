"""One validation case flown by several models and by SAROPS (web runs eval-<split>-s3-k<k>-...): the track over the particles at the
launch and at the end of the flight (fast turns in red), and the planning POS over the flight. For a case whose score jumps between
evaluations.

    python -m rl_search.scripts.rl_eval.case_tracks --split bval --k 0 --runs "160만=r10_2map-1600000" "200만=r10_2map-2000000" \
        --out rl_search/data/real/amboy/rl_eval/report
Writes <out>/case_<split>_k<k>.png.
"""
from __future__ import annotations

import argparse
import base64
import gzip
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from rl_search.scripts.rl_eval.train_report import INK, MUTED, korean_font, style

KEY = "realtime_t5_d3z5l15_s75_o150_i0.75_a0_w1.5"
TN = {"road": "도로형", "mixed": "혼합형", "desert": "사막형"}
SPLIT_NAME = {"nval": "암보이 nval", "bval": "바스토 bval"}


def flight(d: Path) -> dict:
    r = json.load(gzip.open(d/"run.json.gz", "rt"))
    meta, u, res = r["meta"], r["opt"]["uavs"][0], r["res"]
    t = np.frombuffer(base64.b64decode(u["log_t"]), np.float32).astype(float)
    xyy = np.frombuffer(base64.b64decode(u["log_xyyaw"]), np.float32).reshape(-1, 3).astype(float)
    w = np.abs(np.degrees(np.gradient(np.unwrap(xyy[:, 2]), t)))
    T, N = res["T"], res["N"]
    pos = np.frombuffer(base64.b64decode(res["pos"]), "<i2").reshape(T, N, 2)
    st = np.frombuffer(base64.b64decode(res["state"]), np.uint8).reshape(T, N)
    ci = meta["case_info"]
    k0 = min(int(round(ci["t_launch_s"]/res["dt_s"])), T - 1)
    k1 = min(int(round(t[-1]/res["dt_s"])), T - 1)
    P = [pos[k][st[k] != 4].astype(float)*res["pos_unit_m"]/1000 for k in (k0, k1)]
    return dict(xy=xyy[:, :2]/1000, w=w, t=t, P0=P[0], P1=P[1], hold=float(meta["holdout"]["pos"]), curve=np.array(meta["pos"], float),
                ci=ci, fast=float((np.gradient(t)*(w > 1.49)).sum()/60))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="bval")
    ap.add_argument("--k", type=int, required=True, help="case index (0-based; the figure shows k + 1)")
    ap.add_argument("--runs", nargs="+", required=True, help='"name=<web run label>" of the RL flights, e.g. "200만=r10_2map-2000000"')
    ap.add_argument("--out", type=Path, default=Path("rl_search/data/real/amboy/rl_eval/report"))
    args = ap.parse_args()
    from rl_search.scripts.real_map.vehicle_web import RUNS      # (sets a Mac font: the Korean font is chosen after it)
    korean_font()
    sets = [("SAROPS", RUNS/f"eval-{args.split}-s3-k{args.k}-sarops-{KEY}")]
    sets += [(n, RUNS/f"eval-{args.split}-s3-k{args.k}-rl-{lab}") for n, lab in (s.split("=", 1) for s in args.runs)]
    F = [(n, flight(d)) for n, d in sets]
    ci = F[0][1]["ci"]
    allxy = np.vstack([np.vstack([f["xy"], f["P0"], f["P1"]]) for _, f in F])
    lo, hi = np.percentile(allxy, 1, axis=0) - 1.5, np.percentile(allxy, 99, axis=0) + 1.5

    fig, axes = plt.subplots(1, len(F) + 1, figsize=(5.2*(len(F) + 1), 5.6), layout="constrained")
    for a, (n, f) in zip(axes, F):
        a.scatter(f["P0"][:, 0], f["P0"][:, 1], s=1.5, color="#c9772e", alpha=.35, lw=0, zorder=2)
        a.scatter(f["P1"][:, 0], f["P1"][:, 1], s=1.5, color="#7a52c7", alpha=.25, lw=0, zorder=2)
        seg = np.stack([f["xy"][:-1], f["xy"][1:]], axis=1)
        a.add_collection(LineCollection(seg, colors=np.where(f["w"][1:] > 1.49, "#d6336c", "#2a78d6"), lw=1.2, zorder=3))
        a.plot(*f["xy"][0], "o", color=INK, ms=6, zorder=4)
        a.plot(*(np.array(ci["lkp"])/1000), "*", color=INK, ms=12, zorder=4)
        a.set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]), aspect="equal", xlabel="동쪽 (km)", ylabel="북쪽 (km)",
              title=f"{n}: 별도 particle POS {f['hold']:.2f}\n1.49°/s보다 빨리 돈 시간 {f['fast']:.0f}분")
        style(a)
    a = axes[-1]
    colors = [MUTED, "#2a78d6", "#d6336c", "#1baf7a", "#7a52c7"]
    for (n, f), c in zip(F, colors):
        x = np.linspace(0, f["t"][-1] - ci["t_launch_s"], len(f["curve"]))/60
        a.plot(x, f["curve"], color=c, lw=2, label=f"{n} ({f['hold']:.2f})")
    a.set(xlabel="이륙 뒤 시간 (분)", ylabel="계획 particle 누적 POS", title="POS가 쌓이는 과정 (괄호: 별도 particle 채점)", ylim=(0, None))
    a.legend(frameon=False, fontsize=9, loc="upper left")
    style(a)
    fig.suptitle(f"{SPLIT_NAME.get(args.split, args.split)} {args.k + 1}번 {TN.get(ci['typ'], ci['typ'])}: 주황 = 이륙 때 particle, 보라 = 비행 끝 "
                 f"particle, 별 = LKP, 검은 점 = 출발, 빨강 = 빠른 선회", fontsize=12, color=INK)
    args.out.mkdir(parents=True, exist_ok=True)
    p = args.out/f"case_{args.split}_k{args.k}.png"
    fig.savefig(p, dpi=95)
    print("saved", p, "|", ", ".join(f"{n} {f['hold']:.3f}" for n, f in F))


if __name__ == "__main__":
    main()

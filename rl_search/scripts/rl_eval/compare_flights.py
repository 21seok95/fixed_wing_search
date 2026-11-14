"""Two sets of validation flights side by side (web runs eval-<split>-s3-k<k>-rl-<label>, e.g. R10 at one step and R9 scored on the
same split with eval_rl): hold-out POS by type with SAROPS, minutes faster than 1.49 deg/s and full circles by type, case by case POS,
and the tracks of the desert cases (particles at the launch, fast turns in red).

    python -m rl_search.scripts.rl_eval.compare_flights --split nval --a "R9 400만=amboy_r9-4000000" --b "R10 200만=amboy_r10-2000000" \
        --out rl_search/data/real/amboy/rl_eval/report
"""
from __future__ import annotations

import argparse
import base64
import gzip
import json
import re
import warnings
from pathlib import Path

import numpy as np

TYPES = ("road", "mixed", "desert")
TN = {"road": "도로형", "mixed": "혼합형", "desert": "사막형"}
TC = {"road": "#2a78d6", "mixed": "#eb6834", "desert": "#1baf7a"}


def shapes(split: str, label: str) -> dict:
    """Per case: hold-out POS, minutes faster than 1.49 deg/s, full circles one way (as export_run.flight_shape), track and particles."""
    from rl_search.scripts.real_map.vehicle_web import RUNS
    out = {}
    for d in RUNS.glob(f"eval-{split}-s3-k*-rl-{label}"):
        m = re.match(rf"eval-{split}-s3-k(\d+)-rl-{re.escape(label)}$", d.name)
        if not m or not (d/"run.json.gz").exists():
            continue
        r = json.load(gzip.open(d/"run.json.gz", "rt"))
        meta, u, res = r["meta"], r["opt"]["uavs"][0], r["res"]
        t = np.frombuffer(base64.b64decode(u["log_t"]), np.float32).astype(float)
        xyy = np.frombuffer(base64.b64decode(u["log_xyyaw"]), np.float32).reshape(-1, 3).astype(float)
        psi = np.unwrap(xyy[:, 2])
        w = np.degrees(np.gradient(psi, t))
        dt = np.gradient(t)
        s = np.sign(w)*(np.abs(w) > 1.0)
        loops, i = 0, 0
        while i < len(s):
            if s[i] == 0:
                i += 1
                continue
            j = i
            while j + 1 < len(s) and s[j + 1] == s[i]:
                j += 1
            loops += int(abs(np.degrees(psi[j] - psi[i]))//360)
            i = j + 1
        T, N = res["T"], res["N"]
        kt = min(int(round(meta["case_info"]["t_launch_s"]/res["dt_s"])), T - 1)
        P = np.frombuffer(base64.b64decode(res["pos"]), "<i2").reshape(T, N, 2)[kt].astype(float)*res["pos_unit_m"]/1000
        S = np.frombuffer(base64.b64decode(res["state"]), np.uint8).reshape(T, N)[kt]
        k = int(m.group(1))
        out[k] = dict(typ=meta["case_info"].get("typ", TYPES[k % 3]), pos=float(meta["holdout"]["pos"]), key=meta["holdout"].get("ref_key"),
                      fast=float((dt*(np.abs(w) > 1.49)).sum()/60), loops=loops, xy=xyy[:, :2]/1000, w=np.abs(w), P=P[S != 4])
    return out


def main():
    warnings.filterwarnings("ignore")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from rl_search.rl_real_eval import sarops_refs
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="nval")
    ap.add_argument("--a", required=True, help='"name=<web run label>" of the first set (e.g. the older model)')
    ap.add_argument("--b", required=True, help='"name=<web run label>" of the second set')
    ap.add_argument("--out", type=Path, required=True, help="folder of the two pictures")
    args = ap.parse_args()
    plt.rcParams["font.family"] = ["AppleGothic", "NanumGothic", "Noto Sans CJK KR", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False
    INK, MUTED, GRID, BG = "#1f1f1e", "#6b6b67", "#e4e3de", "#fcfcfb"
    (na, la), (nb, lb) = (args.a.split("=", 1), args.b.split("=", 1))
    A, B = shapes(args.split, la), shapes(args.split, lb)
    ks = sorted(set(A) & set(B))
    if not ks:
        raise SystemExit(f"no common cases: {la} {len(A)}, {lb} {len(B)}")
    key = B[ks[0]]["key"]
    try:
        sar = {k: r["holdout"]["pos"] for (st, k), r in sarops_refs(args.split, key).items() if st == 3}
    except FileNotFoundError:
        sar = {}
    col = {na: "#1f1f1e", nb: "#d6336c", "SAROPS": "#9a9a95"}
    fig, axs = plt.subplots(1, 3, figsize=(18, 5.4), facecolor=BG)
    for ax in axs:
        ax.set_facecolor(BG); ax.grid(color=GRID, lw=.6, axis="y"); [sp.set_color(GRID) for sp in ax.spines.values()]
        ax.tick_params(colors=MUTED, labelsize=9)
    x = np.arange(3)
    sets = ([("SAROPS", {k: dict(pos=v, typ=TYPES[k % 3]) for k, v in sar.items() if k in ks})] if sar else []) + [(na, A), (nb, B)]
    wbar = .8/len(sets)
    for j, (name, D) in enumerate(sets):
        v = [np.mean([D[k]["pos"] for k in ks if k in D and TYPES[k % 3] == t]) for t in TYPES]
        b = axs[0].bar(x + (j - (len(sets) - 1)/2)*wbar, v, wbar*.92, color=col[name], label=name)
        for bb, val in zip(b, v):
            axs[0].text(bb.get_x() + bb.get_width()/2, val + .008, f"{val:.3f}", ha="center", fontsize=8.5, color=INK)
    axs[0].set_title(f"(가) 검증 {len(ks)}개 hold-out POS, 유형별 평균", loc="left", color=INK, fontsize=10.5)
    wbar = .4
    for j, (name, D) in enumerate([(na, A), (nb, B)]):
        v = [np.mean([D[k]["fast"] for k in ks if TYPES[k % 3] == t]) for t in TYPES]
        lp = [np.mean([D[k]["loops"] for k in ks if TYPES[k % 3] == t]) for t in TYPES]
        b = axs[1].bar(x + (j - .5)*wbar, v, wbar*.92, color=col[name], label=name)
        for bb, val, l in zip(b, v, lp):
            axs[1].text(bb.get_x() + bb.get_width()/2, val + .4, f"{val:.1f}\n({l:.1f})", ha="center", fontsize=8.5, color=INK)
    axs[1].set_title("(나) 1.49°/s보다 빨리 돈 분 (60분 중), 괄호 = 360° 회전 수", loc="left", color=INK, fontsize=10.5)
    for ax in axs[:2]:
        ax.set_xticks(x); ax.set_xticklabels([TN[t] for t in TYPES], fontsize=10, color=INK); ax.legend(fontsize=8.5, frameon=False)
    ax = axs[2]; ax.grid(color=GRID, lw=.6)
    for k in ks:
        t = TYPES[k % 3]
        ax.scatter(A[k]["pos"], B[k]["pos"], s=46, color=TC[t], edgecolor=BG, lw=1.2, zorder=3, label=TN[t] if k < 3 else None)
        ax.annotate(str(k + 1), (A[k]["pos"], B[k]["pos"]), xytext=(5, -3), textcoords="offset points", fontsize=7.5, color=MUTED)
    hi = max(max(A[k]["pos"], B[k]["pos"]) for k in ks)*1.08
    ax.plot([0, hi], [0, hi], color=MUTED, ls="--", lw=1); ax.set_xlim(0, hi); ax.set_ylim(0, hi)
    ax.set_xlabel(na, color=MUTED, fontsize=9); ax.set_ylabel(nb, color=MUTED, fontsize=9)
    ax.set_title(f"(다) 사례별 hold-out POS (선 위 = {nb}가 높음)", loc="left", color=INK, fontsize=10.5)
    ax.legend(fontsize=8.5, frameon=False, loc="lower right")
    fig.suptitle(f"검증 {args.split}: {nb} vs {na}", x=.01, ha="left", color=INK, fontsize=12)
    fig.tight_layout()
    args.out.mkdir(parents=True, exist_ok=True)
    tag = re.sub(r"[^\w.-]+", "_", f"{lb}_vs_{la}")
    p1 = args.out/f"compare_{args.split}_{tag}.png"; fig.savefig(p1, dpi=110, facecolor=BG); print("saved", p1)
    # desert tracks
    g = np.load("rl_search/data/real/amboy/road_graph_25m.npz")
    p_, o_ = g["edge_poly"].astype(float)/1000, g["edge_poly_offset"]
    roads = [p_[o_[i]:o_[i + 1]] for i in range(len(o_) - 1)]
    dk = [k for k in ks if TYPES[k % 3] == "desert"]
    fig, axs = plt.subplots(2, len(dk), figsize=(4.1*len(dk), 8.9), facecolor=BG, layout="constrained", squeeze=False)
    for j, k in enumerate(dk):
        allp = np.vstack([A[k]["P"], A[k]["xy"], B[k]["xy"]])
        lo, hi = np.percentile(allp, 1, axis=0), np.percentile(allp, 99, axis=0)
        mid, half = (lo + hi)/2, max((hi - lo).max()/2 + .8, 2.0)
        for i, (name, D) in enumerate([(na, A), (nb, B)]):
            ax = axs[i, j]; ax.set_facecolor(BG)
            ax.add_collection(LineCollection(roads, colors="#d9d8d3", linewidths=.7, zorder=1))
            ax.scatter(D[k]["P"][:, 0], D[k]["P"][:, 1], s=1.5, color="#c9772e", alpha=.3, lw=0, zorder=2)
            xy, wv = D[k]["xy"], D[k]["w"]
            seg = np.stack([xy[:-1], xy[1:]], axis=1); fast = wv[:-1] > 1.49
            ax.add_collection(LineCollection(seg[~fast], colors="#2a6db0", linewidths=1.1, zorder=3))
            ax.add_collection(LineCollection(seg[fast], colors="#d6336c", linewidths=1.1, zorder=4))
            ax.set_xlim(mid[0] - half, mid[0] + half); ax.set_ylim(mid[1] - half, mid[1] + half); ax.set_aspect("equal")
            ax.tick_params(colors=MUTED, labelsize=7, length=0); [sp.set_color("#d9d8d3") for sp in ax.spines.values()]
            ax.set_title(f"{name} · 사막형 시나리오 {k + 1}\nPOS {D[k]['pos']:.3f}" + (f" (SAROPS {sar[k]:.3f})" if k in sar else "")
                         + f" · 빠른 선회 {D[k]['fast']:.0f}분 · 360° {D[k]['loops']}번", fontsize=9.5, color=INK, loc="left")
    fig.suptitle("사막형 검증: 파랑 = 1.49°/s 이하로 비행, 빨강 = 더 빨리 선회, 주황 = 이륙 시각의 입자, 회색 = 도로", x=.01, ha="left", color=INK, fontsize=11.5)
    p2 = args.out/f"tracks_desert_{args.split}_{tag}.png"; fig.savefig(p2, dpi=105, facecolor=BG); print("saved", p2)
    for name, D in ((na, A), (nb, B)):
        print(name, " | ".join(f"{TN[t]} POS {np.mean([D[k]['pos'] for k in ks if TYPES[k % 3] == t]):.3f} 빠른 선회 "
                               f"{np.mean([D[k]['fast'] for k in ks if TYPES[k % 3] == t]):.1f}분 360° "
                               f"{np.mean([D[k]['loops'] for k in ks if TYPES[k % 3] == t]):.1f}" for t in TYPES))


if __name__ == "__main__":
    main()

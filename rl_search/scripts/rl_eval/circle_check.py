"""Circling and out-of-view belief in validation flights (user 2026-11-16: unneeded circling in the middle and near the end of a flight;
far belief may not show when the UAV is near the terrain edge). Reads the web runs eval-<split>-s3-k<k>-rl-<label> (eval_rl or the
in-training evaluations) and, per flight minute after the hand-over (the rule's transit is left out):

- circling: the heading turned one way by 330 deg or more over the 3 min around the minute (about one circle at the usual 2.5-3 deg/s);
- POS gained in the minute (planner particles);
- remaining belief within 3 km of the UAV, and 3-15 km away but reachable in the time left (33 m/s);
- share of the remaining belief outside the policy's largest map (heading up, +- side / 2).

Prints a table by flight phase and writes <out>/circle_<label>.json, circle_<label>_timeline.png, circle_<label>_need.png.

    python -m rl_search.scripts.rl_eval.circle_check --label r12_2map-6000000 --name "R12 600만" --map-km 51.2 \
        --out rl_search/data/real/amboy/rl_eval/report
R12 600만 (2026-11-16): circling 13 / 21 / 21 / 33 % of the minutes in 0-15 / 15-30 / 30-45 / 45-60 min; 0.8 % of the belief out of view.
"""
from __future__ import annotations

import argparse
import base64
import gzip
import json
import warnings
from pathlib import Path

import numpy as np

TN = {"road": "도로", "mixed": "혼합", "desert": "사막"}
SPN = {"nval": "암보이", "bval": "바스토"}
TC = {"road": "#2a78d6", "mixed": "#eb6834", "desert": "#1baf7a"}
V_UAV = 33.0
PHASES = ((0, 15), (15, 30), (30, 45), (45, 60))


def flight_minutes(run_dir: Path, half_map_m: float) -> tuple[str, list]:
    r = json.load(gzip.open(run_dir/"run.json.gz", "rt"))
    opt, res, meta = r["opt"], r["res"], r["meta"]
    T, N, dt = res["T"], res["N"], res["dt_s"]
    tl = meta["case_info"]["t_launch_s"]
    P = np.frombuffer(base64.b64decode(res["pos"]), "<i2").reshape(T, N, 2).astype(np.float32)*res["pos_unit_m"]
    S = np.frombuffer(base64.b64decode(res["state"]), np.uint8).reshape(T, N)
    W = np.frombuffer(base64.b64decode(opt["weight"]), np.uint8).reshape(opt["T"], N).astype(np.float32)/255
    u = opt["uavs"][0]
    t = np.frombuffer(base64.b64decode(u["log_t"]), np.float32).astype(float)
    xyy = np.frombuffer(base64.b64decode(u["log_xyyaw"]), np.float32).reshape(-1, 3).astype(float)
    psi = np.unwrap(xyy[:, 2])
    t_h = t[0]                                     # hand-over: the first moment 10 % of the particles are within 5 km (rl_real._transit)
    for i in range(0, len(t), 10):
        j = min(int(round(t[i]/dt)), T - 1)
        pts = P[j][S[j] != 4]
        if np.mean(np.hypot(pts[:, 0] - xyy[i, 0], pts[:, 1] - xyy[i, 1]) <= 5000) >= .1:
            t_h = t[i]
            break
    pt, pv = np.array(opt["pos_t"]), np.array(opt["pos"])
    out = []
    for m in range(60):
        t0, t1 = tl + 60*m, tl + 60*(m + 1)
        sel = (t >= t0) & (t < t1)
        if sel.sum() < 10 or t1 <= t_h:
            continue
        win = (t >= t0 - 60) & (t < t1 + 60)
        j = min(int(round(t0/dt)), len(W) - 1, T - 1)
        w = W[j]*(S[j] != 4)
        tot = float(w.sum())
        x, y, yaw = np.interp(t0, t, xyy[:, 0]), np.interp(t0, t, xyy[:, 1]), np.interp(t0, t, psi)
        dx, dy = P[j][:, 0] - x, P[j][:, 1] - y
        d = np.hypot(dx, dy)
        fwd, right = dx*np.cos(yaw) + dy*np.sin(yaw), dx*np.sin(yaw) - dy*np.cos(yaw)
        outside = (np.abs(fwd) > half_map_m) | (np.abs(right) > half_map_m)
        reach = V_UAV*60*(60 - m)
        out.append(dict(m=m, net_turn=float(abs(np.degrees(psi[win][-1] - psi[win][0]))) if win.sum() > 2 else 0.0,
                        abs_turn=float(np.abs(np.diff(np.degrees(psi[sel]))).sum()),
                        gain=float(np.interp(t1, pt, pv) - np.interp(t0, pt, pv)), rem=tot/N,
                        near3=float(w[d <= 3000].sum()/N), far=float(w[(d > 3000) & (d <= min(15000, reach))].sum()/N),
                        outside=float(w[outside].sum()/tot) if tot > 0 else 0.0))
    return meta["case_info"].get("typ", ""), out


def main():
    warnings.filterwarnings("ignore")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.ticker import FuncFormatter
    from rl_search.scripts.real_map.vehicle_web import RUNS
    from rl_search.scripts.rl_eval.train_report import korean_font
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", required=True, help="web run label (eval-<split>-s3-k<k>-rl-<label>)")
    ap.add_argument("--name", default=None, help="name in the pictures (default: the label)")
    ap.add_argument("--splits", nargs="+", default=["nval", "bval"])
    ap.add_argument("--map-km", type=float, default=51.2, help="side (km) of the model's largest map (R13: 102.4)")
    ap.add_argument("--out", type=Path, default=Path("rl_search/data/real/amboy/rl_eval/report"))
    args = ap.parse_args()
    korean_font()
    plt.rcParams["axes.unicode_minus"] = False
    INK, MUTED, GRID, BG = "#1f1f1e", "#6b6b67", "#e4e3de", "#fcfcfb"
    name = args.name or args.label
    per, rows = {}, []
    for sp in args.splits:
        for k in range(15):
            d = RUNS/f"eval-{sp}-s3-k{k}-rl-{args.label}"
            if not (d/"run.json.gz").exists():
                continue
            typ, mins = flight_minutes(d, args.map_km*500.0)
            per[(sp, k)] = (typ, mins)
            rows += [dict(sp=sp, k=k, typ=typ, **mm) for mm in mins]
    if not rows:
        raise SystemExit(f"no web runs for {args.label}")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/f"circle_{args.label}.json").write_text(json.dumps(rows))
    circ = lambda r: r["net_turn"] >= 330
    print(f"{name}: {len(per)} flights, {len(rows)} flight minutes after the hand-over")
    print("phase    circling | POS per min circling / other | circling: within 3 km / 3-15 km reachable | out of view (mean, minutes > 1 %)")
    for a, b in PHASES:
        rr = [r for r in rows if a <= r["m"] < b]
        cc, oo = [r for r in rr if circ(r)], [r for r in rr if not circ(r)]
        f = lambda L, key: np.mean([r[key] for r in L]) if L else float("nan")
        print(f"{a:2d}-{b:2d} min  {len(cc)/max(len(rr), 1):6.1%} | {f(cc, 'gain'):.4f} / {f(oo, 'gain'):.4f} | {f(cc, 'near3'):.3f} / {f(cc, 'far'):.3f} | "
              f"{f(rr, 'outside'):.4f}, {np.mean([r['outside'] > .01 for r in rr]):.1%}")
    keys = sorted(per, key=lambda c: (c[0] != "nval", ["road", "mixed", "desert"].index(per[c][0]) if per[c][0] in TN else 3, c[1]))
    labels = [f"{SPN.get(c[0], c[0])} {c[1] + 1}번 {TN.get(per[c][0], '')}" for c in keys]
    A = np.full((len(keys), 60), np.nan); G = np.full((len(keys), 60), np.nan); O = np.full((len(keys), 60), np.nan)
    C = np.zeros((len(keys), 60), bool)
    for i, c in enumerate(keys):
        for mm in per[c][1]:
            A[i, mm["m"]], G[i, mm["m"]], O[i, mm["m"]] = mm["abs_turn"], max(mm["gain"], 1e-5), max(mm["outside"], 1e-4)
            C[i, mm["m"]] = mm["net_turn"] >= 330
    fmt = FuncFormatter(lambda v, _: f"{v:g}")
    fig, axs = plt.subplots(1, 4, figsize=(25, 0.36*len(keys) + 2.8), facecolor=BG, gridspec_kw=dict(width_ratios=[1, 1, 1, .9]))
    yy, xx = np.nonzero(C)
    for ax, M, cmap, norm, title in ((axs[0], A, "magma_r", None, "(가) 분당 선회량 (°/분, 점 = 원 선회)"),
                                     (axs[1], G, "Greens", LogNorm(1e-4, 5e-2), "(나) 분당 POS 증가 (로그)"),
                                     (axs[2], O, "magma_r", LogNorm(1e-4, 0.3), f"(다) 남은 확률 중 {args.map_km:g} km 지도 밖 몫")):
        im = ax.imshow(M, cmap=cmap, norm=norm, vmin=None if norm else 0, vmax=None if norm else 240, aspect="auto", interpolation="nearest")
        ax.scatter(xx, yy, s=8, color="#35c2ff" if ax is axs[0] else "#d6336c", lw=0)
        ax.set_yticks(range(len(keys))); ax.set_yticklabels(labels if ax is axs[0] else [], fontsize=8.5, color=INK)
        ax.set_xlabel("이륙 뒤 분", color=MUTED); ax.set_title(title, loc="left", color=INK, fontsize=10.5)
        cb = fig.colorbar(im, ax=ax, fraction=.03, pad=.01); cb.ax.tick_params(labelsize=7.5)
        if norm is not None:
            cb.ax.yaxis.set_major_formatter(fmt)
    ax = axs[3]; ax.set_facecolor(BG); ax.grid(color=GRID, lw=.6); [s.set_color(GRID) for s in ax.spines.values()]
    for typ in TN:
        share = []
        for m in range(60):
            rr = [r for r in rows if r["typ"] == typ and r["m"] == m]
            share.append(np.mean([circ(r) for r in rr]) if rr else np.nan)
        ax.plot(np.arange(60), np.convolve(np.nan_to_num(share), np.ones(5)/5, mode="same")*100, color=TC[typ], lw=2, label=f"{TN[typ]}형")
    ax.set_xlabel("이륙 뒤 분", color=MUTED); ax.set_ylabel("원 선회한 비행의 비율 (%, 5분 이동 평균)", color=MUTED)
    ax.set_title("(라) 시간대별 원 선회 비율", loc="left", color=INK, fontsize=10.5); ax.legend(fontsize=9, frameon=False)
    fig.suptitle(f"{name} · 검증 비행을 1분씩: 선회, POS 증가, 지도 밖 확률 (원 선회 = 앞뒤 3분 동안 한 방향 330° 이상)", x=.01, ha="left", color=INK, fontsize=12.5)
    fig.tight_layout(); p = args.out/f"circle_{args.label}_timeline.png"; fig.savefig(p, dpi=100, facecolor=BG); print("saved", p)
    fig, axs = plt.subplots(1, 2, figsize=(17, 7.2), facecolor=BG)
    for ax in axs:
        ax.set_facecolor(BG); ax.grid(color=GRID, lw=.6); [s.set_color(GRID) for s in ax.spines.values()]; ax.tick_params(colors=MUTED, labelsize=9)
    ax = axs[0]
    for r in rows:
        if circ(r):
            ax.scatter(max(r["near3"], 1e-4), max(r["far"], 1e-4), s=12 + 3000*r["gain"], color=plt.cm.viridis(r["m"]/60), alpha=.75, lw=0)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.xaxis.set_major_formatter(fmt); ax.yaxis.set_major_formatter(fmt)
    ax.plot((1e-4, 1), (1e-4, 1), color=MUTED, ls="--", lw=1); ax.set_xlim(1e-4, 1); ax.set_ylim(1e-4, 1)
    ax.set_xlabel("원 선회한 분의 시작, 3 km 안에 남은 확률", color=MUTED); ax.set_ylabel("같은 순간, 3~15 km(남은 시간에 갈 수 있는 곳)에 남은 확률", color=MUTED)
    ax.set_title("(가) 원 선회한 분들 (색 = 이륙 뒤 분, 크기 = 그 분의 POS 증가, 선 위 = 다른 곳에 더 많음)", loc="left", color=INK, fontsize=10)
    fig.colorbar(plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(0, 60)), ax=ax, fraction=.04).set_label("이륙 뒤 분", color=MUTED)
    ax = axs[1]; x = np.arange(len(PHASES)); wb = .38
    gc = [np.mean([r["gain"] for r in rows if a <= r["m"] < b and circ(r)] or [np.nan]) for a, b in PHASES]
    go = [np.mean([r["gain"] for r in rows if a <= r["m"] < b and not circ(r)] or [np.nan]) for a, b in PHASES]
    for off, v, col, lab in ((-wb/2, gc, "#d6336c", "원 선회한 분"), (wb/2, go, "#2a6db0", "나머지 분")):
        bars = ax.bar(x + off, v, wb*.92, color=col, label=lab)
        for bb, val in zip(bars, v):
            ax.text(bb.get_x() + bb.get_width()/2, val*1.02, f"{val:.4f}", ha="center", fontsize=8.5, color=INK)
    ax.set_xticks(x); ax.set_xticklabels([f"{a}~{b}분" for a, b in PHASES], color=INK); ax.set_ylabel("분당 POS 증가 (평균)", color=MUTED)
    ax.legend(fontsize=9, frameon=False); ax.set_title("(나) 시간대별 분당 POS 증가", loc="left", color=INK, fontsize=10.5)
    fig.suptitle(f"{name} · 원 선회는 필요했나", x=.01, ha="left", color=INK, fontsize=12.5)
    fig.tight_layout(); p = args.out/f"circle_{args.label}_need.png"; fig.savefig(p, dpi=100, facecolor=BG); print("saved", p)


if __name__ == "__main__":
    main()

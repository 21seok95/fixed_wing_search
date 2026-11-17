"""Checks of the redesigned scenario pools (rl_eval_set.V2, user 2026-11-17): what the files hold, and the RL env drawing its training starts
from them (launch at the report time 10-40 km from the LKP). Figure: per 120 km map the LKPs of the pool by type, the particles of two
scenarios at 30 / 60 min, the launch points of 40 training resets, closed land (military, cities).

    python -m rl_search.scripts.rl_eval.pool_v2_check
"""
from __future__ import annotations

import json
import pickle
import warnings
from pathlib import Path

import numpy as np

REGIONS = ("amboy120", "mojave120")
NAME = {"amboy120": "암보이 120 km", "mojave120": "바스토 120 km"}
OUT = Path("rl_search/data/real/amboy/rl_eval/report")
TCOL = {"road": "#2a78d6", "mixed": "#eb6834", "desert": "#1baf7a"}
TN = {"road": "도로형", "mixed": "혼합형", "desert": "사막형"}


def main():
    warnings.filterwarnings("ignore")
    from rl_search.rl_real import RealRLConfig, RealRLEnv, pool_dir_multi, preload
    from rl_search.scripts.rl_eval.signal_check import settings_of
    base = settings_of(json.loads(Path("rl_search/data/rl_real/r12_2map/config.json").read_text()))
    res, draw = {}, {}
    for i_r, r in enumerate(REGIONS):
        terr = preload(r)["terr"]
        n_ax = len(terr.axis)

        def closed_at(xy):
            """'military', 'city' or '' at a point (nearest 25 m cell)."""
            i, j = (int(np.clip(np.round((v - terr.axis[0])/terr.cell), 0, n_ax - 1)) for v in (xy[1], xy[0]))
            mil = getattr(terr, "military", None)
            return "military" if mil is not None and mil[i, j] else "city" if terr.city[i, j] else ""
        d = Path(pool_dir_multi({"scen_v2": True}, r))
        files = sorted(d.glob("*_s*.pkl"))
        rows = []
        scens = []
        for f in files:
            with open(f, "rb") as fh:
                x = pickle.load(fh)
            sc = x["scen"]
            k60 = int(round(3600/sc.save_dt_s))
            out60 = float(((sc.exit_step >= 0) & (sc.exit_step <= k60)).mean())
            rows.append(dict(file=f.name, type=x.get("type"), mb=f.stat().st_size/1e6, steps=int(sc.track.shape[0]), particles=int(sc.track.shape[1]),
                             minutes=(sc.track.shape[0] - 1)*sc.save_dt_s/60, exits_move=bool(getattr(sc, "exits_move", False)),
                             lkp_km=[float(v)/1000 for v in sc.lkp], out60=out60, fast_kmh=x["vc"].get("fast_kmh"),
                             exit_straight=x["vc"].get("exit_straight"), no_return=x["vc"].get("no_return")))
            scens.append((x.get("type"), sc))
        settings = dict(base) | dict(scen_v2=True)
        # launches only here (stage 3); each map its own seed (the same seed drew the same launch distances on both maps: 11-17 fix)
        env = RealRLEnv(RealRLConfig(region=r, multi=True, pool=str(d), settings=settings, stage=3, stage_mix={3: 1.0}), seed=3 + i_r)
        starts = []
        for e in range(40):
            obs, info = env.reset()
            sc = env.env.scen
            st = np.asarray(sc.start, float)[:2]
            starts.append(dict(start_km=(st/1000).tolist(), lkp_km=(np.asarray(sc.lkp, float)[:2]/1000).tolist(),
                               dist_km=float(np.hypot(*(st - np.asarray(sc.lkp, float)[:2]))/1000), t_launch_s=float(env.env.launch_track[0][0]),
                               typ=getattr(env, "_typ", ""), closed=closed_at(st)))
        for _ in range(20):
            obs, rwd, term, trunc, info = env.step(np.zeros(1, np.float32))
        res[r] = dict(files=rows, starts=starts, steps_ok=True, obs_keys=sorted(obs))
        draw[r] = dict(scens=scens, starts=starts, terr=terr)
        print(r, f"{len(rows)} files, {np.mean([x['mb'] for x in rows]):.1f} MB each, {rows[0]['minutes']:.0f} min tracks, exits keep moving: "
                 f"{all(x['exits_move'] for x in rows)}; launch distance {min(s['dist_km'] for s in starts):.1f}-{max(s['dist_km'] for s in starts):.1f} km, "
                 f"launch times {sorted({s['t_launch_s'] for s in starts})}; launches in military land {sum(s['closed'] == 'military' for s in starts)}, "
                 f"in cities {sum(s['closed'] == 'city' for s in starts)} of {len(starts)}")
    (OUT/"pool_v2_check.json").write_text(json.dumps(res, indent=1))
    figure(res, draw)


def figure(res, draw):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["AppleGothic", "NanumGothic", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False
    INK, MUTED, GRID, BG = "#1f1f1e", "#6b6b67", "#e4e3de", "#fcfcfb"
    fig, axs = plt.subplots(1, 3, figsize=(24, 8.2), facecolor=BG, gridspec_kw=dict(width_ratios=[1, 1, 0.8]))
    for ax, r in zip(axs[:2], REGIONS):
        t = draw[r]["terr"]
        a = t.axis[::8]/1000
        for m, col in ((getattr(t, "military", None), "#6b7a3a"), (t.city, "#8b1e2d")):
            if m is not None and m.any():
                ax.contourf(a, a, m[::8, ::8].astype(float), levels=[.5, 2], colors=[col], alpha=0.25)
        shown = 0
        for typ, sc in draw[r]["scens"]:
            if shown < 2 and typ in ("road", "mixed"):
                for k_min, al in ((30, 0.15), (60, 0.35)):
                    k = int(round(k_min*60/sc.save_dt_s))
                    P = sc.track[k, ::5].astype(float)/1000
                    ax.scatter(P[:, 0], P[:, 1], s=1.2, color=TCOL[typ], alpha=al, lw=0)
                shown += 1
        for typ, sc in draw[r]["scens"]:
            ax.plot(*(np.asarray(sc.lkp, float)[:2]/1000), marker="*", ms=13, color=TCOL[typ], mec="white", mew=0.8, zorder=5)
        for s in draw[r]["starts"]:
            ax.plot([s["start_km"][0], s["lkp_km"][0]], [s["start_km"][1], s["lkp_km"][1]], color=INK, lw=0.5, alpha=0.35)
            closed = s.get("closed")
            ax.plot(*s["start_km"], marker="^", ms=7 if closed else 6, color="white" if closed else INK, mec="#8b1e2d" if closed else "white",
                    mew=1.4 if closed else 0.5, zorder=4)
        n_mil = sum(s.get("closed") == "military" for s in draw[r]["starts"])
        n_city = sum(s.get("closed") == "city" for s in draw[r]["starts"])
        ax.text(0.01, 0.01, f"속이 빈 세모 = 군사 구역 안 이륙 {n_mil}번, 도시 안 {n_city}번 (40번 중)", transform=ax.transAxes, fontsize=9, color="#8b1e2d")
        ax.plot([-15, 15, 15, -15, -15], [-15, -15, 15, 15, -15], color=INK, lw=1, ls="--")
        ax.set_xlim(-60, 60); ax.set_ylim(-60, 60); ax.set_aspect("equal")
        ax.set_title(f"{NAME[r]}: ★ = 저장분의 LKP (색 = 유형), 점 = 시나리오 둘의 입자(연함 30분, 진함 60분)\n▲ = 학습 이륙점 40번 (선 = LKP까지), 점선 = LKP 범위(중앙 30 km), "
                     f"올리브·빨강 = 군사 구역·큰 도시", loc="left", fontsize=10, color=INK)
        ax.set_xlabel("동쪽 (km)", fontsize=9, color=MUTED); ax.set_ylabel("북쪽 (km)", fontsize=9, color=MUTED)
        ax.tick_params(colors=MUTED, labelsize=8.5)
    ax = axs[2]
    ax.hist([[s["dist_km"] for s in res[r]["starts"]] for r in REGIONS], bins=np.arange(8, 44, 2), color=["#2a78d6", "#eb6834"],
            label=[f"{NAME[r]} (이륙 시각 모두 신고 시각)" for r in REGIONS])
    ax.axvline(10, color=INK, lw=1, ls="--"); ax.axvline(40, color=INK, lw=1, ls="--")
    ax.set_title("학습 이륙점에서 LKP까지 거리 (km): 정한 범위 10~40 km", loc="left", fontsize=10.5, color=INK)
    ax.set_xlabel("km", fontsize=9, color=MUTED); ax.set_ylabel("이륙 수", fontsize=9, color=MUTED)
    ax.legend(fontsize=8.5, frameon=False)
    ax.grid(color=GRID, lw=.6); ax.tick_params(colors=MUTED, labelsize=8.5)
    for a_ in axs:
        a_.set_facecolor(BG)
        for s_ in a_.spines.values():
            s_.set_color(GRID)
    rows = [x for r in REGIONS for x in res[r]["files"]]
    fig.suptitle(f"새 시나리오 저장분 확인: 지도마다 {len(res[REGIONS[0]]['files'])}개 시험 생성, 파일 하나 {np.mean([x['mb'] for x in rows]):.0f} MB, "
                 f"궤적 {rows[0]['minutes']:.0f}분, 지도 밖 입자는 계속 움직임 · 학습 환경이 저장분에서 읽어 10~40 km에서 바로 이륙",
                 x=0.01, ha="left", fontsize=12.5, color=INK)
    fig.tight_layout()
    fig.savefig(OUT/"pool_v2_check.png", dpi=85, facecolor=BG)


if __name__ == "__main__":
    main()

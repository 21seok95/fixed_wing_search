"""Checks of the stretched maps (rl_search.stretch, user 2026-11-17: the 60 km maps with every speed and length halved = the same maps
read with every length doubled, so the RL env, UAV, camera and reward code stay the original):

  layers - every terrain layer of amboy_x2 / mojave_x2 at a point 2p equals the base map's at p (height, slope, closed land, roads,
           water, forest, city, the RL env's passable grid, the camera visibility); slope must stay the real one
  spread - the particles made on the base map at half the redesigned speeds and doubled, against the redesigned scenarios at full speed on
           the real 120 km maps (spread120_check.json): R95 around the LKP and the share outside the map, per minute
  flight - R12 (6M) flies one redesigned scenario on each stretched map (launch 10-40 km in the stretched metres = 5-20 km real)
  figure - report/stretch_check.png and .md

    python -m rl_search.scripts.rl_eval.stretch_check layers
    python -m rl_search.scripts.rl_eval.stretch_check spread --n 12 --workers 6
    python -m rl_search.scripts.rl_eval.stretch_check flight
    python -m rl_search.scripts.rl_eval.stretch_check figure
"""
from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import replace
from multiprocessing import get_context
from pathlib import Path

import numpy as np

REGIONS = ("amboy_x2", "mojave_x2")
NAME = {"amboy_x2": "암보이 60 km × 2", "mojave_x2": "바스토 60 km × 2"}
REAL120 = {"amboy_x2": "amboy120", "mojave_x2": "mojave120"}
TYPES = ("road", "mixed", "desert")
TN = {"road": "도로형", "mixed": "혼합형", "desert": "사막형"}
OUT = Path("rl_search/data/real/amboy/rl_eval/report")
WORK = OUT/"stretch"
MODEL = "rl_search/data/rl_real/r12_2map/ckpt/model_6000000.zip"
MINUTES = np.arange(0, 71, 2)


def layers():
    warnings.filterwarnings("ignore")
    from rl_search.detection import DetectionConfig, load_landcover, shared_visibility
    from rl_search.rl_env import RLConfig, passable_grid
    from rl_search.rl_real import preload
    res = {}
    for r in REGIONS:
        st = preload(r)
        T, B = st["terr"], preload(st["base"])["terr"]
        k = st["k"]
        rng = np.random.default_rng(0)
        n = len(B.axis)
        i, j = rng.integers(0, n, 20000), rng.integers(0, n, 20000)
        p = np.stack([B.axis[j], B.axis[i]], 1)                     # base cell centres (x, y)
        q = p*k                                                     # the same points on the stretched map

        def at(t, xy):
            jj = np.clip(np.rint((xy[:, 0] - t.axis[0])/t.cell).astype(int), 0, len(t.axis) - 1)
            ii = np.clip(np.rint((xy[:, 1] - t.axis[0])/t.cell).astype(int), 0, len(t.axis) - 1)
            return ii, jj
        bi, bj = at(B, p)
        ti, tj = at(T, q)
        row = dict(cells_same=bool((bi == ti).all() and (bj == tj).all()), cell_base=float(B.cell), cell_stretched=float(T.cell),
                   half_base_km=float(B.axis[-1]/1000), half_stretched_km=float(T.axis[-1]/1000))
        for name in ("z", "water", "fords", "forest", "stream", "surface", "city", "built", "slope_deg", "forbidden"):
            a, b = getattr(B, name), getattr(T, name)
            row[name] = float(np.mean(a[bi, bj] == b[ti, tj]))
        row["slope_max_diff_deg"] = float(np.abs(B.slope_deg - T.slope_deg).max())
        row["slope_plain_recomputed_deg"] = float(np.degrees(np.arctan(np.hypot(*np.gradient(B.z, T.cell, T.cell)))).max())
        row["slope_real_max_deg"] = float(B.slope_deg.max())
        cfg = DetectionConfig()
        Ab = shared_visibility(B, load_landcover(st["base"], float(B.cell)), cfg)
        At = shared_visibility(T, load_landcover(r, float(T.cell)), cfg)
        row["visibility_same"] = float(np.mean(Ab == At))
        row["visibility_diff_cells"] = int((Ab != At).sum())
        c = RLConfig().cell_m
        pb, pt = passable_grid(B, c), passable_grid(T, c*k)
        row["passable_same"] = float(np.mean(pb == pt)) if pb.shape == pt.shape else None
        res[r] = row
        samp = rng.choice(len(p), 3000, replace=False)
        np.savez_compressed(WORK/f"layers_{r}.npz", slope_b=B.slope_deg[bi[samp], bj[samp]], slope_t=T.slope_deg[ti[samp], tj[samp]],
                            z_b=B.z[bi[samp], bj[samp]], z_t=T.z[ti[samp], tj[samp]])
        print(r, json.dumps(row), flush=True)
    (WORK/"layers.json").write_text(json.dumps(res, indent=1))


_NET = {}


def _init(regions):
    from rl_search.rl_real import preload
    for r in regions:
        _NET[r] = preload(r)


def spread_job(a):
    """spread120_check.job for a stretched map: the same seeds, type rules, route settings, particle rules and 2,000 particles, on the
    base map at the redesigned speeds / k with the LKP in the central 30 km / k, then every length x k."""
    region, typ, i = a
    warnings.filterwarnings("ignore")
    from rl_search.rl_eval_set import DIVERSE, diverse_lkp, multi_vc, route_vc, vehicle_config
    from rl_search.scripts.rl_eval.spread_check import NEW as SPEEDS
    from rl_search.stretch import stretched_vc
    from rl_search.vehicle import NO_BACKTRACK, simulate_vehicle
    st = _NET[region]
    net, k = st["net"], st["k"]
    s = 7_000_000 + 1000*TYPES.index(typ) + i + (500 if region.startswith("mojave") else 0)
    rng = np.random.default_rng(s)
    vc = stretched_vc(multi_vc(rng, typ) | SPEEDS | route_vc(s) | NO_BACKTRACK, k)
    lkp = diverse_lkp(net, rng, DIVERSE["lkp_half_m"]/k)
    heading = float(rng.uniform(0.0, 360.0))
    cfg = replace(replace(vehicle_config(), **vc), particles=2000, duration_s=4200.0, exit_straight=True)
    res = simulate_vehicle(net, cfg, lkp, heading, np.random.default_rng(s))
    tr, dt = res.track.astype(float)*k, float(cfg.save_dt_s)
    L = np.asarray(res.lkp_road, float)*k
    half = float(st["terr"].axis[-1])
    out = dict(region=region, typ=typ, i=i, lkp_cheb_km=float(np.abs(L).max()/1000), outside=[], r50=[], r95=[], need=[])
    for m in MINUTES:
        kk = min(int(round(m*60/dt)), tr.shape[0] - 1)
        P = tr[kk]
        d = np.hypot(*(P - L).T)/1000
        out["outside"].append(float((np.abs(P).max(axis=1) > half).mean()))
        out["r50"].append(float(np.percentile(d, 50)))
        out["r95"].append(float(np.percentile(d, 95)))
        out["need"].append(float(np.abs(L).max()/1000 + np.percentile(np.abs(P - L), 99, axis=0).max()/1000))
    return out


def spread(n: int, workers: int):
    warnings.filterwarnings("ignore")
    _init(REGIONS)
    jobs = [(r, t, i) for r in REGIONS for t in TYPES for i in range(n)]
    with get_context("fork").Pool(workers) as pool:
        rows = pool.map(spread_job, jobs, chunksize=1)
    (WORK/"spread.json").write_text(json.dumps(rows))


def flight():
    warnings.filterwarnings("ignore")
    from rl_search.rl_real import RealRLConfig, RealRLEnv
    from rl_search.rl_real_eval import _model
    from rl_search.scripts.rl_eval.signal_check import settings_of
    model = _model(MODEL)
    settings = settings_of(json.loads(Path("rl_search/data/rl_real/r12_2map/config.json").read_text())) | dict(scen_v2=True)
    out = {}
    for i, r in enumerate(REGIONS):
        env = RealRLEnv(RealRLConfig(region=r, multi=True, pool=None, settings=settings, stage=3, stage_mix={3: 1.0}), seed=21 + i)
        env._pick_type = lambda: "road"                         # one road-type scenario per map (the widest spread)
        obs, info = env.reset()
        e = env.env
        sc = e.scen
        path = [env.transit_start[1:3]] + [(x, y) for _, x, y, _ in env.transit_track]
        pos, t0 = [], e.t
        term = trunc = False
        v = []
        while not (term or trunc):
            p0 = e.plant.pose[:2].copy()
            a, _ = model.predict(obs, deterministic=True)
            obs, rw, term, trunc, info = env.step(a)
            path.append(tuple(e.plant.pose[:2]))
            pos.append(float(info["POS"]))
            v.append(float(np.hypot(*(e.plant.pose[:2] - p0))/5.0))
        k30 = int(round(30*60/sc.save_dt_s))
        out[r] = dict(path=np.array(path).tolist(), pos=pos, lkp=np.asarray(sc.lkp, float).tolist(), start=np.asarray(sc.start, float).tolist(),
                      launch_km=float(np.hypot(*(np.asarray(sc.start) - np.asarray(sc.lkp)))/1000), speed_mps=float(np.median(v)),
                      handover_min=float((t0 - 0)/60), final_pos=pos[-1] if pos else 0.0)
        np.save(WORK/f"flight_{r}_p30.npy", sc.track[k30, ::3].astype(np.float32))
        print(r, f"launch {out[r]['launch_km']:.1f} km (stretched), UAV {out[r]['speed_mps']:.1f} m/s (stretched), POS {out[r]['final_pos']:.3f}",
              flush=True)
    (WORK/"flight.json").write_text(json.dumps(out))


def figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["AppleGothic", "NanumGothic", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False
    INK, MUTED, GRID, BG = "#1f1f1e", "#6b6b67", "#e4e3de", "#fcfcfb"
    TC = {"road": "#2a78d6", "mixed": "#eb6834", "desert": "#1baf7a"}
    lay = json.loads((WORK/"layers.json").read_text())
    sp = json.loads((WORK/"spread.json").read_text())
    real = json.loads((OUT/"spread120_check.json").read_text())
    fl = json.loads((WORK/"flight.json").read_text()) if (WORK/"flight.json").exists() else {}
    fig = plt.figure(figsize=(24, 14.5), facecolor=BG)
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.15])
    # 1. layers
    ax = fig.add_subplot(gs[0, 0])
    names = ["z", "slope_deg", "forbidden", "surface", "water", "forest", "city", "built", "passable_same", "visibility_same"]
    lab = ["고도", "경사", "못 가는 땅", "도로", "물", "숲", "큰 도시", "마을", "학습 환경의\n지나갈 칸", "카메라 가시도"]
    for i, r in enumerate(REGIONS):
        vals = [100*(lay[r][nm] if lay[r][nm] is not None else np.nan) for nm in names]
        ax.barh(np.arange(len(names)) + (i - 0.5)*0.38, vals, height=0.36, color=("#2a78d6", "#eb6834")[i], label=NAME[r])
        for y, v in zip(np.arange(len(names)) + (i - 0.5)*0.38, vals):
            ax.text(min(v, 100) + 0.3, y, f"{v:.2f}%" if v < 100 else "100%", va="center", fontsize=7.5, color=INK)
    ax.set_yticks(range(len(names)), lab, fontsize=8.5); ax.invert_yaxis()
    ax.set_xlim(95, 101.5); ax.set_xlabel("대응 지점에서 원래 60 km 지도와 같은 값 (%)", fontsize=8.5, color=MUTED)
    ax.set_title("확인 1: 늘린 지도의 층 = 원래 지도의 층 (2만 지점)", loc="left", fontsize=10.5, color=INK)
    ax.legend(fontsize=8, frameon=False, loc="lower left")
    ax = fig.add_subplot(gs[0, 1])
    for i, r in enumerate(REGIONS):
        d = np.load(WORK/f"layers_{r}.npz")
        ax.scatter(d["slope_b"], d["slope_t"], s=4, alpha=0.4, color=("#2a78d6", "#eb6834")[i], label=f"{NAME[r]}: 늘린 지도에 넣은 경사")
        ax.scatter(d["slope_b"], np.degrees(np.arctan(np.tan(np.radians(d["slope_b"]))/2)), s=3, alpha=0.25, color="#9a9a95",
                   label="다시 계산했다면 (가로 2배라 절반)" if i == 0 else None)
    ax.plot([0, 45], [0, 45], color=INK, lw=0.8, ls="--")
    ax.axhline(32, color="#d6336c", lw=0.8, ls=":"); ax.text(1, 32.6, "못 가는 경사 32°", fontsize=8, color="#d6336c")
    ax.set_xlabel("원래 지도의 경사 (°)", fontsize=8.5, color=MUTED); ax.set_ylabel("늘린 지도의 경사 (°)", fontsize=8.5, color=MUTED)
    ax.set_title("경사는 실제 값 그대로 (다시 계산하지 않음)", loc="left", fontsize=10.5, color=INK)
    ax.legend(fontsize=7.8, frameon=False, loc="upper left")
    # 2. spread
    for col, key, ttl in ((2, "r95", "확인 2: 입자 95%가 드는 반경 (km)"), (3, "outside", "지도 밖 입자 (%)")):
        ax = fig.add_subplot(gs[0, col])
        for t in TYPES:
            for src, ls, nm in ((sp, "-", "60 km × 2, 절반 속도"), (real, "--", "실제 120 km, 원래 속도")):
                rows = [x for x in src if x["typ"] == t]
                if not rows:
                    continue
                y = np.median([x[key] for x in rows], axis=0)*(100 if key == "outside" else 1)
                ax.plot(MINUTES, y, color=TC[t], ls=ls, lw=1.8, label=f"{TN[t]}: {nm}")
        ax.set_xlabel("LKP 뒤 분", fontsize=8.5, color=MUTED)
        ax.set_title(ttl + "\n두 지도 합쳐 유형마다 24개의 중앙값 (실선 늘린 지도, 점선 실제 120 km)", loc="left", fontsize=10.5, color=INK)
        if col == 2:
            ax.legend(fontsize=7.5, frameon=False, ncol=1)
    # 3. flights
    for col, r in enumerate(REGIONS):
        if r not in fl:
            continue
        ax = fig.add_subplot(gs[1, col*2:col*2 + 2])
        f = fl[r]
        P = np.load(WORK/f"flight_{r}_p30.npy")/1000
        ax.scatter(P[:, 0], P[:, 1], s=1.2, color="#c9772e", alpha=0.35, lw=0, label="입자 (30분)")
        p = np.array(f["path"])/1000
        ax.plot(p[:, 0], p[:, 1], color=INK, lw=0.9, label=f"R12 비행 (찾은 POS {f['final_pos']:.2f})")
        ax.plot(*p[0], marker="^", ms=10, color=INK, mec="white", ls="none", label=f"이륙점 (LKP에서 {f['launch_km']:.0f} km = 실제 {f['launch_km']/2:.0f} km)")
        ax.plot(*np.array(f["lkp"])/1000, marker="*", ms=14, color="#d6336c", mec="white", ls="none", label="LKP")
        ax.plot([-60, 60, 60, -60, -60], [-60, -60, 60, 60, -60], color=GRID, lw=1)
        ax.set_aspect("equal"); ax.set_xlim(-62, 62); ax.set_ylim(-62, 62)
        sec = ax.secondary_xaxis("top", functions=(lambda x: x/2, lambda x: x*2))
        sec.set_xlabel("실제 거리 (km)", fontsize=8.5, color=MUTED); sec.tick_params(colors=MUTED, labelsize=8)
        ax.set_xlabel("늘린 지도의 거리 (km) = 학습 환경이 보는 값", fontsize=8.5, color=MUTED)
        ax.set_title(f"확인 3: {NAME[r]} 도로형 시나리오를 R12가 비행: 기체 {f['speed_mps']:.1f} m/s(늘린 지도) = 실제 {f['speed_mps']/2:.1f} m/s",
                     loc="left", fontsize=10.5, color=INK)
        ax.legend(fontsize=8, frameon=False, loc="lower left")
    for a in fig.axes:
        a.set_facecolor(BG); a.tick_params(colors=MUTED, labelsize=8)
        for s in a.spines.values():
            s.set_color(GRID)
    fig.suptitle("늘린 지도 확인: 60 km 지도를 모든 길이 2배로 읽음 (= 속도·길이 절반 세상). 무인기·카메라·보상 코드는 원래 그대로",
                 x=0.01, ha="left", fontsize=13.5, color=INK)
    fig.tight_layout()
    fig.savefig(OUT/"stretch_check.png", dpi=80, facecolor=BG)
    L = ["# 늘린 지도 확인", "", "| 지도 | 칸 위치 같음 | " + " | ".join(lab).replace("\n", " ") + " | 가시도 다른 칸 |", "|---|---|" + "---|"*(len(lab) + 1)]
    for r in REGIONS:
        L.append(f"| {NAME[r]} | {lay[r]['cells_same']} | " + " | ".join(f"{100*lay[r][nm]:.2f}%" if lay[r][nm] is not None else "-" for nm in names)
                 + f" | {lay[r]['visibility_diff_cells']} |")
    L += ["", "| 유형 | R95 60분: 늘린 지도 / 실제 120 km (km) | 지도 밖 60분: 늘린 / 실제 (%) |", "|---|---|---|"]
    i60 = int(np.flatnonzero(MINUTES == 60)[0])
    for t in TYPES:
        a = [x for x in sp if x["typ"] == t]
        b = [x for x in real if x["typ"] == t]
        L.append(f"| {TN[t]} | {np.median([x['r95'][i60] for x in a]):.1f} / {np.median([x['r95'][i60] for x in b]):.1f} | "
                 f"{100*np.mean([x['outside'][i60] for x in a]):.1f} / {100*np.mean([x['outside'][i60] for x in b]):.1f} |")
    (OUT/"stretch_check.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=("layers", "spread", "flight", "figure"))
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    if a.what == "layers":
        layers()
    elif a.what == "spread":
        spread(a.n, a.workers)
    elif a.what == "flight":
        flight()
    else:
        figure()


if __name__ == "__main__":
    main()

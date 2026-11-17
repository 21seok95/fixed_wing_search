"""The 120 km maps (amboy120, mojave120; transverse Mercator, elevation float32; user 2026-11-17) against the 60 km maps they extend.

- overlay: the 60 km maps' roads moved into the new frame (old x, y -> latitude, longitude by the old projection -> new x, y) over the
  new roads, on the whole 120 km map and in a 12 km window;
- elevation: the new terrain at the old grid points (through latitude / longitude) minus the old elevation;
- statistics: road length (fast / slow, the network the particles use), dead-end share, land shares (forest, water, built, city, steep
  ground), for the 60 km map, the same area inside the 120 km map, and the whole 120 km map;
- memory of the 120 km maps (the slim terrain of 2026-11-17) and a smoke test: particles with the agreed rules (vehicle.NO_BACKTRACK)
  and the new speeds on each 120 km map, and an RL env (R12's settings) reset and flown 30 decisions there.

    python -m rl_search.scripts.real_map.map120_check
"""
from __future__ import annotations

import json
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates

PAIRS = (("amboy", "amboy120"), ("mojave", "mojave120"))
OUT = Path("rl_search/data/real/amboy/rl_eval/report")
NAME = {"amboy": "암보이", "mojave": "바스토"}


def to_new(r60, r120, x, y):
    from rl_search.real_terrain import to_latlon, to_xy
    la, lo = to_latlon(r60, x, y)
    return to_xy(r120, la, lo)


def to_old(r60, r120, x, y):
    from rl_search.real_terrain import to_latlon, to_xy
    la, lo = to_latlon(r120, x, y)
    return to_xy(r60, la, lo)


def sample(terr, fld, x, y):
    rows = (np.asarray(y) - terr.axis[0])/terr.cell
    cols = (np.asarray(x) - terr.axis[0])/terr.cell
    return map_coordinates(fld, [rows.ravel(), cols.ravel()], order=1, mode="nearest").reshape(np.shape(x))


def stats(net, area_mask=None, edge_in=None):
    """Land shares on the grid cells of ``area_mask`` (every 2nd cell) and road numbers of the edges ``edge_in``."""
    t = net.terr
    sl = (slice(None, None, 2), slice(None, None, 2))
    m = np.ones(t.z[sl].shape, bool) if area_mask is None else area_mask
    land = ~(t.water | t.fords)[sl]
    steep = t.steep(t.cfg.max_slope_deg)[sl] & land
    e_in = np.ones(len(net.edge_len), bool) if edge_in is None else edge_in
    main = net.edge_main & e_in
    area_km2 = m.sum()*(2*t.cell)**2/1e6
    mil = t.military[sl][m].mean() if getattr(t, "military", None) is not None else 0.0
    return dict(area_km2=float(area_km2), forest=float(t.forest[sl][m].mean()), water=float(t.water[sl][m].mean()), military=float(mil),
                built=float(t.built[sl][m].mean()), city=float(t.city[sl][m].mean()), steep=float(steep[m].mean()),
                fast_km=float(net.edge_len[main & (net.edge_tier == 2)].sum()/1000), slow_km=float(net.edge_len[main & (net.edge_tier == 1)].sum()/1000),
                deadend=float(net.edge_len[main & ~net.edge_core].sum()/max(net.edge_len[main].sum(), 1)), edges=int(main.sum()))


def check_pair(k60, k120):
    from rl_search.real_terrain import REGIONS
    from rl_search.vehicle import load_net
    r60, r120 = REGIONS[k60], REGIONS[k120]
    n60, n120 = load_net(k60), load_net(k120)
    t60, t120 = n60.terr, n120.terr
    out = dict(cells=len(t120.axis), z_dtype=str(t120.z.dtype))
    # roads of the 60 km map in the new frame
    px, py = to_new(r60, r120, n60.poly[:, 0].astype(float), n60.poly[:, 1].astype(float))
    poly_old = np.column_stack([px, py])
    s = np.linspace(-30000, 30000, 121)
    sq = np.concatenate([np.column_stack([s, np.full_like(s, -30000)]), np.column_stack([np.full_like(s, 30000), s]),
                         np.column_stack([s[::-1], np.full_like(s, 30000)]), np.column_stack([np.full_like(s, -30000), s[::-1]])])
    sqn = np.column_stack(to_new(r60, r120, sq[:, 0], sq[:, 1]))
    # how far each old road vertex is from the nearest new road cell (both from OSM; differences: projection, OSM edits since, simplification)
    from scipy.ndimage import distance_transform_edt
    road_new = n120.cell_edge >= 0
    dist_new = distance_transform_edt(~road_new)*t120.cell
    used_old = np.zeros(len(n60.poly), bool)
    for e in np.flatnonzero(n60.edge_main):
        used_old[n60.off[e]:n60.off[e + 1]] = True
    d_old = sample(t120, dist_new.astype(np.float32), poly_old[used_old, 0], poly_old[used_old, 1])
    out["old_road_to_new_road_m"] = dict(p50=float(np.percentile(d_old, 50)), p90=float(np.percentile(d_old, 90)), p99=float(np.percentile(d_old, 99)),
                                         share_within_25m=float((d_old <= 25).mean()))
    # elevation: new terrain at the old grid points
    a = t60.axis[::6]
    X, Y = np.meshgrid(a, a)
    xn, yn = to_new(r60, r120, X, Y)
    dz = sample(t120, t120.z, xn, yn) - t60.z[::6, ::6]
    out["dz_m"] = dict(p1=float(np.percentile(dz, 1)), p50=float(np.percentile(dz, 50)), p99=float(np.percentile(dz, 99)), abs_max=float(np.abs(dz).max()))
    # statistics: the 60 km map, the same area in the 120 km map, the whole 120 km map
    a2 = t120.axis[::2]
    X2, Y2 = np.meshgrid(a2, a2)
    xo, yo = to_old(r60, r120, X2, Y2)
    same = (np.abs(xo) <= 30000) & (np.abs(yo) <= 30000)
    mid = np.asarray(n120.poly[(n120.off[:-1] + n120.off[1:])//2], float)
    mxo, myo = to_old(r60, r120, mid[:, 0], mid[:, 1])
    e_same = (np.abs(mxo) <= 30000) & (np.abs(myo) <= 30000)
    out["stats"] = {"60 km 지도": stats(n60), "120 km 지도의 같은 구역": stats(n120, same, e_same), "120 km 지도 전체": stats(n120)}
    out["memory_mb"] = dict(terrain=sum(v.nbytes for v in vars(t120).values() if isinstance(v, np.ndarray))/1e6,
                            road_grid=sum(v.nbytes for k, v in vars(n120).items() if isinstance(v, np.ndarray) and v.ndim == 2 and k != "out")/1e6)
    draw = dict(n60=n60, n120=n120, poly_old=poly_old, used_old=used_old, sqn=sqn, dz=dz, X=xn, Y=yn)
    return out, draw


def smoke(k120):
    """Particles with the agreed rules and an RL env on the 120 km map."""
    from rl_search.rl_eval_set import vehicle_config
    from rl_search.rl_real import RealRLConfig, RealRLEnv, preload
    from rl_search.real_scenario import VehicleScenarios
    from rl_search.scripts.rl_eval.signal_check import settings_of
    from rl_search.scripts.rl_eval.spread_check import NEW as SPEEDS
    from rl_search.vehicle import NO_BACKTRACK, simulate_vehicle
    st = preload(k120)
    net = st["net"]
    cfg = replace(vehicle_config(), particles=2000, duration_s=4200.0, **SPEEDS, **NO_BACKTRACK)
    res = simulate_vehicle(net, cfg, np.array([2000.0, -3000.0]), 40.0, np.random.default_rng(1), diag=True)
    k60 = int(round(3600/cfg.save_dt_s))
    gone = float(((res.exit_step >= 0) & (res.exit_step <= k60)).mean())
    d = np.hypot(*(res.track[k60].astype(float) - res.lkp_road).T)/1000
    out = dict(particles_exit60=gone, r50_km=float(np.percentile(d, 50)), r95_km=float(np.percentile(d, 95)))
    settings = settings_of(json.loads(Path("rl_search/data/rl_real/r12_2map/config.json").read_text()))
    env = RealRLEnv(RealRLConfig(region=k120, settings=dict(settings)), seed=0)
    vs = VehicleScenarios(st["sampler"], replace(vehicle_config(), **SPEEDS, **NO_BACKTRACK), region=k120, net=net)
    scen = vs.make_at(5, np.array([2000.0, -3000.0]), 40.0, start=(-25000.0, -20000.0))
    obs, info = env.reset(options=dict(scenario=scen, stage=3))
    pos = [float(info["POS"])]
    for _ in range(30):
        obs, r, term, trunc, info = env.step(np.zeros(1, np.float32))
        pos.append(float(info["POS"]))
    out["rl_env"] = dict(obs_shapes={k: list(v.shape) for k, v in obs.items()} if isinstance(obs, dict) else list(obs.shape),
                         half_km=float(env.half/1000), pos_after_30=pos[-1])
    return out


def figure(results, draws):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import LightSource, LinearSegmentedColormap
    plt.rcParams["font.family"] = ["AppleGothic", "NanumGothic", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False
    INK, MUTED, GRID, BG = "#1f1f1e", "#6b6b67", "#e4e3de", "#fcfcfb"
    div = LinearSegmentedColormap.from_list("d", ["#2a78d6", "#f2f1ed", "#eb6834"])
    fig, axs = plt.subplots(2, 4, figsize=(26, 13.5), facecolor=BG, gridspec_kw=dict(width_ratios=[1.25, 1, 1, 1.1]))
    for row, (k60, k120) in enumerate(PAIRS):
        res, dr = results[k120], draws[k120]
        n120, n60 = dr["n120"], dr["n60"]
        t = n120.terr
        a = t.axis/1000
        s = 4
        hs = LightSource(azdeg=315, altdeg=45).hillshade(np.asarray(t.z[::s, ::s], float), vert_exag=2, dx=t.cell*s, dy=t.cell*s)
        segs = lambda net, tier: [net.poly[net.off[e]:net.off[e + 1]]/1000 for e in np.flatnonzero(net.edge_main & (net.edge_tier == tier))]
        old_lines = [dr["poly_old"][n60.off[e]:n60.off[e + 1]]/1000 for e in np.flatnonzero(n60.edge_main)]
        for col, lim in ((0, None), (1, 6.0)):
            ax = axs[row, col]
            ax.imshow(hs, origin="lower", extent=[a[0], a[-1], a[0], a[-1]], cmap="gray", vmin=-0.4, vmax=1.3, alpha=0.55)
            ax.add_collection(LineCollection(segs(n120, 1), colors="#b9905f", linewidths=0.5 if lim is None else 2.2))
            ax.add_collection(LineCollection(segs(n120, 2), colors="#3b3a37", linewidths=0.8 if lim is None else 2.8))
            ax.add_collection(LineCollection(old_lines, colors="#d6336c", linewidths=0.35 if lim is None else 1.0))
            ax.plot(dr["sqn"][:, 0]/1000, dr["sqn"][:, 1]/1000, color="#2a78d6", lw=1.4, ls="--")
            if getattr(t, "military", None) is not None:              # military land: closed to the targets
                ax.contourf(a[::4], a[::4], t.military[::4, ::4].astype(float), levels=[.5, 2], colors=["#6b7a3a"], alpha=0.35)
                ax.contour(a[::4], a[::4], t.military[::4, ::4].astype(float), levels=[.5], colors=["#4a5526"], linewidths=1.0)
            if getattr(t, "city", None) is not None and t.city.any():
                ax.contour(a[::4], a[::4], t.city[::4, ::4].astype(float), levels=[.5], colors=["#8b1e2d"], linewidths=0.9)
            if lim is None:
                ax.set_xlim(a[0], a[-1]); ax.set_ylim(a[0], a[-1])
                ax.set_title(f"{NAME[k60]} 120 km: 도로 (진회색 빠른 길, 갈색 느린 길), 분홍 = 60 km 지도의 도로를 새 좌표로 옮긴 것, 파란 점선 = 60 km 지도의 범위\n"
                             f"올리브 = 군사 구역, 짙은 빨강 테두리 = 큰 도시 (둘 다 입자가 못 들어감)",
                             loc="left", fontsize=9.5, color=INK)
            else:
                ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
                st = res["old_road_to_new_road_m"]
                ax.set_title(f"가운데 12 km: 60 km 지도의 도로 점에서 새 도로까지 중앙값 {st['p50']:.0f} m, 90% {st['p90']:.0f} m\n"
                             f"(25 m 안 {100*st['share_within_25m']:.0f}%; 차이는 도로 단순화와 OSM 갱신)", loc="left", fontsize=9.5, color=INK)
            ax.set_aspect("equal")
            ax.set_xlabel("동쪽 (km)", fontsize=8.5, color=MUTED); ax.set_ylabel("북쪽 (km)", fontsize=8.5, color=MUTED)
        ax = axs[row, 2]
        dz = dr["dz"]
        lim = max(float(np.percentile(np.abs(dz), 99.5)), 0.5)
        im = ax.imshow(dz, origin="lower", extent=[-30, 30, -30, 30], cmap=div, vmin=-lim, vmax=lim)
        fig.colorbar(im, ax=ax, shrink=0.75, label="m")
        z = res["dz_m"]
        ax.set_title(f"고도 차이: 새 120 km 지형 − 60 km 지형 (60 km 지도 좌표)\n1~99%: {z['p1']:+.1f} ~ {z['p99']:+.1f} m, 중앙값 {z['p50']:+.2f} m",
                     loc="left", fontsize=9.5, color=INK)
        ax.set_xlabel("동쪽 (km, 예전 좌표)", fontsize=8.5, color=MUTED); ax.set_ylabel("북쪽 (km, 예전 좌표)", fontsize=8.5, color=MUTED)
        ax = axs[row, 3]
        S = res["stats"]
        names = list(S)
        cols = ("#9a9a95", "#2a78d6", "#0b2f5e")
        items = (("fast_km", "빠른 길 (km/100 km²)"), ("slow_km", "느린 길 (km/100 km²)"), ("forest", "숲 (%)"), ("built", "시가지·마을 (%)"),
                 ("city", "큰 도시 (%)"), ("military", "군사 구역 (%)"), ("steep", "경사 32° 이상 (%)"), ("deadend", "막다른 가지 (도로 길이 %)"))
        y = np.arange(len(items))
        for j, nm in enumerate(names):
            v = []
            for k, _ in items:
                x = S[nm][k]
                v.append(x/S[nm]["area_km2"]*100 if k in ("fast_km", "slow_km") else 100*x)
            ax.barh(y + (j - 1)*0.27, v, height=0.25, color=cols[j], label=f"{nm} ({S[nm]['area_km2']:.0f} km²)")
            for yy, vv in zip(y + (j - 1)*0.27, v):
                ax.text(vv + 0.3, yy, f"{vv:.1f}", va="center", fontsize=7.5, color=INK)
        ax.set_yticks(y, [lab for _, lab in items], fontsize=9, color=INK); ax.invert_yaxis()
        ax.set_title(f"{NAME[k60]}: 도로와 땅 (입자가 쓰는 도로망 기준)", loc="left", fontsize=10.5, color=INK)
        ax.legend(fontsize=8, frameon=False, loc="lower right")
        for a_ in axs[row]:
            a_.set_facecolor(BG); a_.tick_params(colors=MUTED, labelsize=8)
            for sp in a_.spines.values():
                sp.set_color(GRID)
    fig.suptitle("120 km 지도 (횡메르카토르, 25 m 칸 4,801 × 4,801)와 지금 60 km 지도의 비교", x=0.01, ha="left", fontsize=14, color=INK)
    fig.tight_layout()
    fig.savefig(OUT/"map120_check.png", dpi=80, facecolor=BG)


def main():
    warnings.filterwarnings("ignore")
    results, draws = {}, {}
    for k60, k120 in PAIRS:
        res, dr = check_pair(k60, k120)
        res["smoke"] = smoke(k120)
        results[k120], draws[k120] = res, dr
        print(k120, json.dumps({k: v for k, v in res.items() if k != "stats"}, indent=1, ensure_ascii=False))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/"map120_check.json").write_text(json.dumps(results, indent=1, ensure_ascii=False))
    L = ["# 120 km 지도 확인", "", "| 지도 | 구역 | 넓이 (km²) | 빠른 길 (km) | 느린 길 (km) | 막다른 가지 | 숲 | 시가지·마을 | 큰 도시 | 군사 구역 | 경사 32° 이상 |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for k60, k120 in PAIRS:
        for nm, s in results[k120]["stats"].items():
            L.append(f"| {NAME[k60]} | {nm} | {s['area_km2']:.0f} | {s['fast_km']:.0f} | {s['slow_km']:.0f} | {100*s['deadend']:.1f}% | {100*s['forest']:.2f}% | "
                     f"{100*s['built']:.2f}% | {100*s['city']:.2f}% | {100*s['military']:.2f}% | {100*s['steep']:.2f}% |")
    (OUT/"map120_check.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    figure(results, draws)


if __name__ == "__main__":
    main()

"""Checks of the transverse Mercator map frame (rl_search.geo, user 2026-11-17) and how it differs from the local equirectangular
projection of the 60 km maps (x = dlon * 111320 cos(lat0), y = dlat * 111320, rl_search.real_terrain.to_xy before).

- round trip: latitude, longitude -> map -> latitude, longitude on a grid over the 120 km square (metres of error);
- meridian arc: on the centre meridian the projected y must equal the meridian arc length, integrated numerically from the radius of
  curvature of the WGS84 ellipsoid;
- a published UTM value: the CN Tower, 43 38'33.24"N 79 23'13.7"W, UTM zone 17, 630 084 m E, 4 833 438 m N (Wikipedia, "Universal
  Transverse Mercator coordinate system");
- length errors of both projections over the 120 km square (east-west and north-south, from the ellipsoid's radii of curvature), and
  how far a point moves between the old and the new map coordinates.

    python -m rl_search.scripts.real_map.projection_check
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from rl_search.geo import LocalTM, meridian_radii, tm_forward, utm

OUT = Path("rl_search/data/real/amboy/rl_eval/report")
CENTRES = {"amboy": (34.56, -115.74), "mojave": (34.95, -116.85)}
HALF = 60000.0


def old_xy(lat, lon, lat0, lon0):
    kx, ky = 111320.0*math.cos(math.radians(lat0)), 111320.0
    return (np.asarray(lon) - lon0)*kx, (np.asarray(lat) - lat0)*ky


def main():
    from scipy.integrate import quad
    res = {}
    # published UTM value
    lat = 43 + 38/60 + 33.24/3600
    lon = -(79 + 23/60 + 13.7/3600)
    zone, E, N = utm(lat, lon)
    res["cn_tower"] = dict(zone=zone, easting=float(E), northing=float(N), expected=[17, 630084, 4833438],
                           err_m=[float(E - 630084), float(N - 4833438)])
    # meridian arc
    arc = []
    for la in (10.0, 34.56, 34.95, 60.0):
        m_num = quad(lambda p: float(meridian_radii(np.degrees(p))[0]), 0, math.radians(la), epsabs=1e-9, epsrel=1e-14)[0]
        arc.append(dict(lat=la, numerical_m=m_num, tm_m=float(tm_forward(la, 0.0, 0.0)[1]), err_mm=1000*(float(tm_forward(la, 0.0, 0.0)[1]) - m_num)))
    res["meridian_arc"] = arc
    fields = {}
    for reg, (lat0, lon0) in CENTRES.items():
        tm = LocalTM(lat0, lon0)
        a = np.linspace(-HALF, HALF, 121)
        X, Y = np.meshgrid(a, a)
        LA, LO = tm.to_latlon(X, Y)
        X2, Y2 = tm.to_xy(LA, LO)
        rt = float(np.hypot(X2 - X, Y2 - Y).max())
        # point scale of the new frame (numerical) and of the old projection, east-west and north-south
        M, Nv = meridian_radii(LA)
        d = 1e-6
        xe, ye = tm.to_xy(LA, LO + d)
        k_tm = np.hypot(xe - X2, ye - Y2)/(Nv*np.cos(np.radians(LA))*math.radians(d))
        kx_old = 111320.0*math.cos(math.radians(lat0))*(180/math.pi)/(Nv*np.cos(np.radians(LA)))
        ky_old = 111320.0*(180/math.pi)/M
        xo, yo = old_xy(LA, LO, lat0, lon0)
        disp = np.hypot(xo - X, yo - Y)
        inner = (np.abs(X) <= 30000) & (np.abs(Y) <= 30000)
        conv = tm.convergence_deg(LA, LO)
        res[reg] = dict(round_trip_max_m=rt, tm_scale_err_pct_max=float(100*np.abs(k_tm - 1).max()),
                        old_ew_err_pct=[float(100*(kx_old - 1).min()), float(100*(kx_old - 1).max())],
                        old_ns_err_pct=[float(100*(ky_old - 1).min()), float(100*(ky_old - 1).max())],
                        move_max_m_120=float(disp.max()), move_max_m_60=float(disp[inner].max()),
                        convergence_deg_max=float(np.abs(conv).max()))
        fields[reg] = dict(X=X, Y=Y, k_tm=k_tm, kx_old=kx_old, ky_old=ky_old, xo=xo, yo=yo, disp=disp)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/"projection_check.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    figure(res, fields)


def figure(res, fields):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    plt.rcParams["font.family"] = ["AppleGothic", "NanumGothic", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False
    INK, MUTED, GRID, BG = "#1f1f1e", "#6b6b67", "#e4e3de", "#fcfcfb"
    seq = LinearSegmentedColormap.from_list("s", ["#f4f7fb", "#9cc3ee", "#2a78d6", "#0b2f5e"])
    div = LinearSegmentedColormap.from_list("d", ["#2a78d6", "#f2f1ed", "#eb6834"])
    fig, axs = plt.subplots(2, 4, figsize=(24, 12), facecolor=BG)
    for row, reg in enumerate(CENTRES):
        f, r = fields[reg], res[reg]
        ext = [-60, 60, -60, 60]
        panels = ((100*(f["kx_old"] - 1), "예전 투영: 동서 길이 오차 (%)", div, True),
                  (100*(f["ky_old"] - 1), "예전 투영: 남북 길이 오차 (%)", div, True),
                  (100*(f["k_tm"] - 1), "횡메르카토르: 길이 오차 (%)", seq, False),
                  (f["disp"], "같은 지점이 옮겨 가는 거리 (m): 예전 → 새 좌표", seq, False))
        for col, (val, title, cmap, sym) in enumerate(panels):
            ax = axs[row, col]
            if sym:
                lim = float(np.abs(val).max())
                im = ax.imshow(val, origin="lower", extent=ext, cmap=cmap, vmin=-lim, vmax=lim)
            else:
                im = ax.imshow(val, origin="lower", extent=ext, cmap=cmap, vmin=0, vmax=float(val.max()))
            cs = ax.contour(f["X"]/1000, f["Y"]/1000, val, levels=6, colors=INK, linewidths=0.6, alpha=0.6)
            ax.clabel(cs, fontsize=7.5, fmt="%.3g")
            ax.plot([-30, 30, 30, -30, -30], [-30, -30, 30, 30, -30], color=INK, lw=1.2, ls="--")
            if col == 3:
                s = 8
                ax.quiver(f["X"][::s, ::s]/1000, f["Y"][::s, ::s]/1000, (f["X"] - f["xo"])[::s, ::s], (f["Y"] - f["yo"])[::s, ::s],
                          color=INK, alpha=0.7, width=0.003)
            fig.colorbar(im, ax=ax, shrink=0.78)
            ax.set_title(f"{'암보이' if reg == 'amboy' else '바스토'}: {title}", loc="left", fontsize=10.5, color=INK)
            ax.set_xlabel("동쪽 (km)", fontsize=8.5, color=MUTED); ax.set_ylabel("북쪽 (km)", fontsize=8.5, color=MUTED)
            ax.tick_params(colors=MUTED, labelsize=8)
    c = res["cn_tower"]
    arc = max(abs(a["err_mm"]) for a in res["meridian_arc"])
    fig.suptitle("좌표 투영 확인 (점선 = 지금 60 km 지도의 범위). "
                 f"왕복 오차 최대 {max(res[r]['round_trip_max_m'] for r in CENTRES)*1000:.4f} mm, 자오선 길이 오차 최대 {arc:.3f} mm, "
                 f"UTM 예제(CN 타워) 차이 동 {c['err_m'][0]:+.2f} m 북 {c['err_m'][1]:+.2f} m (발표값은 m 단위 반올림)",
                 x=0.01, ha="left", fontsize=12.5, color=INK)
    fig.tight_layout()
    fig.savefig(OUT/"projection_check.png", dpi=85, facecolor=BG)


if __name__ == "__main__":
    main()

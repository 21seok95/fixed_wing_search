"""Why the m3 "step" danger channel spreads on the 6.4 km map (user 2026-11-24: "1 km 밖은 0이라고 했는데 6.4 km 지도에서 퍼짐이 많이 보이는
이유?"): the channel is the max over the teammate's fan points (where it can be within the map's lead: 15 / 30 / 60 s on the 1.6 / 3.2 /
6.4 km maps) of (time value: 1 now -> 0.3 at the lead) x step(distance from that point). Row 1: the teammate's present position only, row 2:
the channel (all fan points), row 3: the values along the line from me through the teammate. Same search moment as report/m3_channels.png.

    python -m rl_search.scripts.multi_design.m3_spread_view
"""
from __future__ import annotations

import pickle

import numpy as np

from rl_search.scripts.multi_design.m3_check import WORK
from rl_search.scripts.multi_design.ma_eval import OUT


def main(case: int = 0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    from rl_search.rl_multi import danger_level
    from rl_search.mate_maps import cell_grid
    plt.rcParams["font.family"] = "AppleGothic"
    plt.rcParams["axes.unicode_minus"] = False
    sn = pickle.load(open(WORK/f"m3_c{case}_snaps.pkl", "rb"))["search"]
    f_m, r_m = sn["mates"][0]
    leads = dict(m1600=15, m3200=30, m6400=60)
    fig, ax = plt.subplots(3, 3, figsize=(15, 14))
    for ci, name in enumerate(("m1600", "m3200", "m6400")):
        mp = sn["maps"][name]
        side, px = mp["side"], mp["step"].shape[0]
        F, R, _, _ = cell_grid(0.0, 0.0, np.pi/2, side, px)          # (forward, right) of the cell centres
        now = danger_level(np.hypot(F - f_m, R - r_m), "step")
        h = side/2000
        for ri, (img, title) in enumerate(((now, "동료 지금 위치만: step(거리)"), (mp["step"], f"채널 (동료가 {leads[name]}초 안에 갈 수 있는 곳 전체)"))):
            a = ax[ri, ci]
            a.imshow(img, cmap="magma", vmin=0, vmax=1, extent=(-h, h, -h, h), origin="upper")
            a.plot(0, 0, marker="^", color="#38b2ac", ms=9)
            a.plot(r_m/1000, f_m/1000, marker="X", color="#63b3ed", ms=9, mec="k")
            for rad, ls in ((0.5, "-"), (1.0, "--")):
                a.add_patch(Circle((r_m/1000, f_m/1000), rad, fill=False, color="#63b3ed", ls=ls, lw=1))
            a.plot([0, r_m/1000*3], [0, f_m/1000*3], color="w", lw=0.8, ls=":")
            a.set_xlim(-h, h)
            a.set_ylim(-h, h)
            a.set_title(f"{side/1000:g} km 지도 — {title}", fontsize=10)
            if ci == 0:
                a.set_ylabel(["① 지금 위치만", "② 실제 채널"][ri])
        # profile along the line me -> teammate (and beyond)
        u = np.array([f_m, r_m])/np.hypot(f_m, r_m)
        s = np.linspace(-side/2, side/2, 400)
        cell = side/px
        rows = np.clip(np.floor((side/2 - s*u[0])/cell).astype(int), 0, px - 1)
        cols = np.clip(np.floor((s*u[1] + side/2)/cell).astype(int), 0, px - 1)
        inside = (np.abs(s*u[0]) <= side/2) & (np.abs(s*u[1]) <= side/2)
        dm = s - np.hypot(f_m, r_m)
        a = ax[2, ci]
        a.plot(dm[inside]/1000, now[rows, cols][inside], color="#d69e2e", label="① 지금 위치만")
        a.plot(dm[inside]/1000, mp["step"][rows, cols][inside], color="#2b6cb0", label="② 실제 채널")
        for x in (-1, -0.5, 0.5, 1):
            a.axvline(x, color="k", lw=0.6, ls="--" if abs(x) == 1 else "-")
        a.set_xlabel("동료 지금 위치에서의 거리 (km, 점선 방향: - 나 쪽, + 그 너머)")
        a.set_ylim(-0.02, 1.05)
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
        a.set_title("③ 나와 동료를 잇는 선(흰 점선) 위의 값", fontsize=10)
    fig.suptitle(f"step 위험 구역 채널 분해 (지리산 사례 {case + 1}, 탐색 중, 동료 {np.hypot(f_m, r_m):.0f} m): 삼각형 = 나, X = 동료 지금 위치, 원 = 동료 지금 위치에서 "
                 "500 m(실선) / 1 km(점선)", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT/"m3_spread.png", dpi=95)
    print(OUT/"m3_spread.png")


if __name__ == "__main__":
    main()

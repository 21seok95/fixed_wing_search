"""Particles that leave the map: kept in the belief (scenarios with ``exits_move``, user 2026-11-17: probability that disappears at the
edge is a wrong signal) or dropped (before). One scenario on the 60 km Barstow map, where a quarter of the particles leave within the
hour (road type, the new speeds and particle rules, exit_straight), flown by R12 (6M) twice: the belief keeping the particles that left
and dropping them. Shows the remaining probability, POS, and the 51.2 km observation map (particles now) at 50 min.

    python -m rl_search.scripts.rl_eval.exit_keep_check
"""
from __future__ import annotations

import json
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np

OUT = Path("rl_search/data/real/amboy/rl_eval/report")
MODEL = "rl_search/data/rl_real/r12_2map/ckpt/model_6000000.zip"


def main():
    warnings.filterwarnings("ignore")
    from rl_search.real_scenario import VehicleScenarios
    from rl_search.rl_eval_set import diverse_lkp, multi_vc, route_vc, vehicle_config
    from rl_search.rl_real import preload
    from rl_search.rl_real_eval import _env, _model
    from rl_search.scripts.rl_eval.signal_check import settings_of
    from rl_search.scripts.rl_eval.spread_check import NEW as SPEEDS
    from rl_search.vehicle import NO_BACKTRACK
    st = preload("mojave")
    s = 7_000_501
    rng = np.random.default_rng(s)
    vc = multi_vc(rng, "road") | SPEEDS | route_vc(s) | NO_BACKTRACK | dict(exit_straight=True, duration_s=4200.0)
    lkp = diverse_lkp(st["net"], rng)
    heading = float(rng.uniform(0.0, 360.0))
    vs = VehicleScenarios(st["sampler"], replace(vehicle_config(), **vc), region="mojave", net=st["net"])
    scen = vs.make_at(s, lkp, heading, start=np.asarray(lkp, float)[:2] + np.array([-15000.0, -12000.0]))
    settings = settings_of(json.loads(Path("rl_search/data/rl_real/r12_2map/config.json").read_text()))
    env, model = _env(settings, "mojave"), _model(MODEL)
    runs = {}
    for keep in (True, False):
        sc = replace(scen, exits_move=keep)
        obs, info = env.reset(options=dict(scenario=sc, stage=3))
        b = env.env.belief
        t0 = env.env.t
        rec = dict(t=[0.0], pos=[float(info["POS"])], mass=[b.mass], gone=[b.escaped], snap=None)
        term = trunc = False
        while not (term or trunc):
            a, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(a)
            m = (env.env.t - t0)/60
            rec["t"].append(m); rec["pos"].append(float(info["POS"])); rec["mass"].append(env.env.belief.mass); rec["gone"].append(env.env.belief.escaped)
            if rec["snap"] is None and env.env.t >= 50*60:
                rec["snap"] = (np.asarray(obs["m51200"]).copy(), env.env.plant.pose.copy(), env.env.t)
        runs[keep] = rec
        print(f"keep={keep}: final POS {rec['pos'][-1]:.3f}, remaining {rec['mass'][-1]:.3f}, counted as gone {rec['gone'][-1]:.3f}")
    left60 = float(((scen.exit_step >= 0) & (scen.exit_step*scen.save_dt_s <= 3600)).mean())
    (OUT/"exit_keep_check.json").write_text(json.dumps({str(k): dict(final_pos=v["pos"][-1], remaining=v["mass"][-1], gone=v["gone"][-1])
                                                        for k, v in runs.items()} | dict(left_map_by_60min=left60)))
    figure(runs, left60)


def figure(runs, left60):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    plt.rcParams["font.family"] = ["AppleGothic", "NanumGothic", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False
    INK, MUTED, GRID, BG = "#1f1f1e", "#6b6b67", "#e4e3de", "#fcfcfb"
    COL = {True: "#2a78d6", False: "#9a9a95"}
    NAME = {True: "남김 (새 시나리오)", False: "뺌 (예전 방식)"}
    seq = LinearSegmentedColormap.from_list("s", ["#fcfcfb", "#9cc3ee", "#2a78d6", "#0b2f5e"])
    fig, axs = plt.subplots(1, 4, figsize=(24, 6.2), facecolor=BG, gridspec_kw=dict(width_ratios=[1.1, 1.1, 1, 1]))
    for keep, r in runs.items():
        axs[0].plot(r["t"], 100*np.array(r["mass"]), color=COL[keep], lw=2.2, label=f"{NAME[keep]}: 남은 확률")
        if not keep:
            axs[0].plot(r["t"], 100*np.array(r["gone"]), color=COL[keep], lw=1.6, ls="--", label=f"{NAME[keep]}: 지도 밖으로 사라진 확률")
        axs[1].plot(r["t"], np.array(r["pos"]), color=COL[keep], lw=2.2, label=NAME[keep])
    axs[0].set_title(f"남은 확률 (%): 이 시나리오는 60분까지 입자의 {100*left60:.0f}%가 지도 밖으로 나감", loc="left", fontsize=11, color=INK)
    axs[1].set_title("찾은 확률 POS (R12, 같은 시나리오)", loc="left", fontsize=11, color=INK)
    for ax in axs[:2]:
        ax.set_xlabel("이륙 뒤 분", fontsize=9, color=MUTED); ax.legend(fontsize=8.5, frameon=False)
        ax.grid(color=GRID, lw=.6); ax.set_facecolor(BG); ax.tick_params(colors=MUTED, labelsize=8.5)
        for s_ in ax.spines.values():
            s_.set_color(GRID)
    for ax, keep in zip(axs[2:], (True, False)):
        m, pose, t = runs[keep]["snap"]
        cnt, vis = m[0], m[-1]
        ax.imshow(cnt, origin="upper", cmap=seq, vmin=0, vmax=max(float(cnt.max()), 1e-6), extent=[-25.6, 25.6, -25.6, 25.6])
        off = np.ma.masked_where(vis > 0, np.ones_like(vis))
        ax.imshow(off, origin="upper", cmap=LinearSegmentedColormap.from_list("o", ["#d9d8d3", "#d9d8d3"]), alpha=0.55,
                  extent=[-25.6, 25.6, -25.6, 25.6])
        ax.plot(0, 0, marker="^", color="#eb6834", ms=10, mec="white")
        ax.set_title(f"{NAME[keep]}: 51.2 km 관측 지도 (지금 입자 위치), {t/60:.0f}분\n회색 = 지도 밖 (가시도 0), 주황 세모 = 기체 (위가 진행 방향)",
                     loc="left", fontsize=10, color=INK)
        ax.set_xlabel("오른쪽 (km)", fontsize=8.5, color=MUTED); ax.set_ylabel("앞쪽 (km)", fontsize=8.5, color=MUTED)
        ax.tick_params(colors=MUTED, labelsize=8)
    fig.suptitle("지도 밖으로 나간 입자: 믿음에 남김(새 시나리오) 대 뺌(예전). 바스토 60 km 도로형, 새 속도·입자 규칙·지도 밖 직진, R12가 비행",
                 x=0.01, ha="left", fontsize=13, color=INK)
    fig.tight_layout()
    fig.savefig(OUT/"exit_keep_check.png", dpi=85, facecolor=BG)


if __name__ == "__main__":
    main()

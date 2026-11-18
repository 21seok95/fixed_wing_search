"""Check of the next run's observation and start (user 2026-11-18, docs/scenario_redesign.md "모델 구조"): seven maps at 2x spacing
(``maps7``), no ground visibility in the observation (``no_vis_obs``), and the policy flying from the launch (``launch_policy``: no rule
transit, so the first decisions see the particles on the large maps only).

The same launches (redesigned scenarios on the stretched maps amboy_x2 / mojave_x2, launch 10-40 km from the LKP at the report time) are
started twice:
  before = R13's layout (five maps 1.6 / 6.4 / 25.6 / 51.2 / 102.4 km, visibility shown, 120 s toward the LKP then the rule transit until
           10 % of the remaining probability is within 5 km): the policy's first decision is at the hand-over
  after  = seven maps, no visibility, the policy's first decision is at the launch
Per map: the share of the remaining probability inside it at the policy's first decision. Per launch: how long the rule flew before.
Also: the observation space and the policy network (RealMapExtractor) of the new layout, a short flight with random actions, and one
observation drawn channel by channel.

    python -m rl_search.scripts.rl_eval.maps7_check --n 12
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np

OUT = Path("rl_search/data/real/amboy/rl_eval/report")
REGIONS = ("amboy_x2", "mojave_x2")


def settings_pair():
    from rl_search.scripts.rl_eval.signal_check import settings_of
    base = settings_of(json.loads(Path("rl_search/data/rl_real/r12_2map/config.json").read_text())) | dict(scen_v2=True)
    before = base | dict(extra_map_m=102400.0, extra_map_lead_s=600.0)                  # R13's five maps
    after = {k: v for k, v in base.items() if k != "handover_km"} | dict(maps7=True, no_vis_obs=True, launch_policy=True)
    return before, after


def make(region, settings, seed):
    from rl_search.rl_real import RealRLConfig, RealRLEnv, pool_dir_multi
    return RealRLEnv(RealRLConfig(region=region, multi=True, pool=pool_dir_multi(settings, region), settings=settings, stage=3), seed=seed)


def shares(env) -> dict:
    """Share of the remaining probability inside each map (heading up, square) now, and the nearest particle (km)."""
    e = env.env
    b = e.belief
    live = b.w > 0
    P, w = b.positions(e.t)[live], b.w[live]
    x, y, yaw = e.plant.pose
    d = P - np.array([x, y])
    c, s = np.cos(yaw), np.sin(yaw)
    fwd, right = d[:, 0]*c + d[:, 1]*s, d[:, 0]*s - d[:, 1]*c
    out = {name: float(w[(np.abs(fwd) < side/2) & (np.abs(right) < side/2)].sum()/w.sum()) for name, side, _ in env.cfg.maps}
    out["near_km"] = float(np.hypot(*d.T).min()/1000.0)
    return out


def run(n: int):
    warnings.filterwarnings("ignore")
    from rl_search.rl_real import make_gym, preload
    sb, sa = settings_pair()
    rows, example = [], None
    for ri, region in enumerate(REGIONS):
        preload(region)
        eb, ea = make(region, sb, 7000 + ri), make(region, sa, 7000 + ri)
        for i in range(n):
            t0 = time.perf_counter()
            scen = eb._new_scenario()
            opts = eb.curriculum_options(scen, 3)
            eb.reset(options=dict(opts))
            hb = shares(eb)
            t_hand = (eb.env.t - float(eb.env.launch_track[0][0]))/60.0
            obs, _ = ea.reset(options=dict(opts))
            ha = shares(ea)
            t_first = (ea.env.t - float(ea.env.launch_track[0][0]))/60.0
            launch_km = float(np.hypot(*(np.asarray(opts["scenario"].start) - np.asarray(scen.lkp))))/1000.0
            rows.append(dict(region=region, i=i, typ=eb._ep_typ, launch_km=launch_km, hand_min=t_hand, first_min=t_first,
                             before=hb, after=ha, pos_rule=float(eb.pos_cum)))
            print(f"{region} {i} {eb._ep_typ:6s} launch {launch_km:4.1f} km, rule {t_hand:4.1f} min; nearest particle at the first decision "
                  f"{hb['near_km']:4.1f} -> {ha['near_km']:4.1f} km  ({time.perf_counter() - t0:.1f} s)", flush=True)
            if example is None and launch_km > 25:
                example = dict(region=region, obs={k: v.copy() for k, v in obs.items()}, typ=eb._ep_typ, launch_km=launch_km,
                               near_km=ha["near_km"])
    # observation space, network, a short flight with random actions (the new layout)
    import torch
    from rl_search.rl_gym import RealMapExtractor
    g = make_gym(make(REGIONS[0], sa, 1).cfg, 1)
    space = g.observation_space
    ext = RealMapExtractor(space)
    o, _ = g.reset(seed=3)
    ok_space = all(space[k].shape == o[k].shape for k in o)
    rng = np.random.default_rng(0)
    t0 = time.perf_counter()
    for _ in range(30):
        o, r, term, trunc, info = g.step(rng.uniform(-1, 1, 1).astype(np.float32))
        ok_space &= all(space[k].contains(o[k]) for k in o if k != "vec")
    step_ms = 1000*(time.perf_counter() - t0)/30
    with torch.no_grad():
        f = ext({k: torch.as_tensor(v)[None] for k, v in o.items()})
    from rl_search.rl_real import RealRLConfig
    old_space = make_gym(RealRLConfig(region=REGIONS[0], settings=sb, multi=True,
                                      pool=make(REGIONS[0], sb, 1).cfg.pool, stage=3), 2).observation_space
    net = dict(shapes={k: list(v.shape) for k, v in space.spaces.items()}, old_shapes={k: list(v.shape) for k, v in old_space.spaces.items()},
               params=sum(p.numel() for p in ext.parameters()), old_params=sum(p.numel() for p in RealMapExtractor(old_space).parameters()),
               feature=list(f.shape), ok_space=bool(ok_space), step_ms=step_ms)
    print(json.dumps(net), flush=True)
    np.savez_compressed(OUT/"maps7_check_example.npz", **{k: v for k, v in example["obs"].items()})
    (OUT/"maps7_check.json").write_text(json.dumps(dict(rows=rows, net=net, example={k: v for k, v in example.items() if k != "obs"})))


def figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from rl_search.rl_real import MAPS7
    plt.rcParams["font.family"] = "AppleGothic"
    plt.rcParams["axes.unicode_minus"] = False
    d = json.loads((OUT/"maps7_check.json").read_text())
    rows, net = d["rows"], d["net"]
    names = [m[0] for m in MAPS7]
    km = [m[1]/1000 for m in MAPS7]
    C_B, C_A = "#8a8f98", "#2a6fdb"
    fig, ax = plt.subplots(1, 3, figsize=(17, 5.2), gridspec_kw=dict(width_ratios=[1.6, 1, 1]))
    # 1: share of the remaining probability inside each map at the policy's first decision
    a = ax[0]
    for j, (nm, k) in enumerate(zip(names, km)):
        for key, col, dx in (("before", C_B, -0.15), ("after", C_A, 0.15)):
            v = np.array([r[key][nm] for r in rows if nm in r[key]])
            if not len(v):
                continue
            q1, med, q3 = np.percentile(v, [25, 50, 75])
            a.plot([j + dx]*2, [q1, q3], color=col, lw=3, solid_capstyle="round")
            a.plot(j + dx, med, "o", color=col, ms=8, mec="white", mew=1.5)
            a.scatter(np.full(len(v), j + dx) + np.random.default_rng(j).uniform(-.05, .05, len(v)), v, s=8, color=col, alpha=.35, lw=0)
    a.set_xticks(range(len(names)), [f"{k:g} km" for k in km])
    a.set_ylabel("지도 안에 든 남은 확률 몫")
    a.set_ylim(-0.03, 1.03)
    a.set_title("정책의 첫 결정 때 지도마다 보이는 확률 (점 = 이륙 하나, 굵은 선 = 사분위, 원 = 중앙값)", fontsize=10)
    a.plot([], [], "o", color=C_B, label="전: R13 지도 5장, 규칙 비행 뒤 넘겨받음")
    a.plot([], [], "o", color=C_A, label="후: 지도 7장, 이륙하자마자")
    a.legend(loc="center right", fontsize=9, frameon=False)
    a.grid(axis="y", color="#e5e5e5", lw=.8)
    # 2: rule flight before the hand-over vs launch distance
    a = ax[1]
    for region, mk in zip(REGIONS, ("o", "s")):
        rr = [r for r in rows if r["region"] == region]
        a.scatter([r["launch_km"] for r in rr], [r["hand_min"] for r in rr], marker=mk, s=30, color=C_B, label=f"전 ({region})")
        a.scatter([r["launch_km"] for r in rr], [r["first_min"] for r in rr], marker=mk, s=30, color=C_A, label=f"후 ({region})")
    a.set_xlabel("이륙점에서 LKP까지 (km)")
    a.set_ylabel("정책이 첫 결정을 하기까지 (분)")
    a.set_title("규칙이 대신 날던 시간", fontsize=10)
    a.legend(fontsize=8, frameon=False)
    a.grid(color="#e5e5e5", lw=.8)
    # 3: nearest particle at the first decision
    a = ax[2]
    for key, col, lab in (("before", C_B, "전"), ("after", C_A, "후")):
        a.scatter([r["launch_km"] for r in rows], [r[key]["near_km"] for r in rows], s=30, color=col, label=lab)
    a.set_xlabel("이륙점에서 LKP까지 (km)")
    a.set_ylabel("첫 결정 때 가장 가까운 입자 (km)")
    a.set_title("첫 결정 때 입자까지 거리", fontsize=10)
    a.legend(fontsize=9, frameon=False)
    a.grid(color="#e5e5e5", lw=.8)
    fig.suptitle(f"관측 지도 7장 + 가시도 없음 + 이륙 직후 인계 (늘린 지도 두 장, 이륙 {len(rows)}번)", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT/"maps7_check.png", dpi=130)
    # example observation, channel by channel
    ex = np.load(OUT/"maps7_check_example.npz")
    chn = ["입자 위치 지금", "입자 위치 예측 시각 뒤", "남은 POS (지도마다 정규화)", "남은 POS (절대)", "카메라 화면 (1.6 km만)"]
    leads = {m[0]: m[2] for m in MAPS7}
    fig, axs = plt.subplots(len(names), len(chn), figsize=(2.3*len(chn), 2.3*len(names)))
    for i, nm in enumerate(names):
        o = ex[nm]
        for j in range(len(chn)):
            a = axs[i, j]
            a.set_xticks([]), a.set_yticks([])
            if j < o.shape[0]:
                a.imshow(o[j], cmap="Blues", vmin=0, vmax=max(float(o[j].max()), 1e-6), interpolation="nearest")
                a.plot(31.5, 31.5, "^", color="#d9480f", ms=5)
            else:
                a.axis("off")
            if i == 0:
                a.set_title(chn[j], fontsize=9)
            if j == 0:
                a.set_ylabel(f"{km[i]:g} km\n(예측 {leads[nm]:g} s)", fontsize=9)
    e = d["example"]
    fig.suptitle(f"이륙 순간의 관측 ({e['region']}, {e['typ']}, 이륙 {e['launch_km']:.1f} km, 가장 가까운 입자 {e['near_km']:.1f} km; "
                 f"위 = 진행 방향, 주황 삼각형 = 무인기)", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT/"maps7_check_obs.png", dpi=110)
    # table
    L = ["# 관측 지도 7장 · 가시도 없음 · 이륙 직후 인계 확인", "",
         f"늘린 지도(amboy_x2, mojave_x2) 이륙 {len(rows)}번. 전 = R13 배치(지도 5장, 가시도, 120초 + 5 km 규칙 비행 뒤 인계), 후 = 지도 7장, 가시도 없음, 이륙 순간 인계.", "",
         "| 지도 | 전: 첫 결정 때 지도 안 확률 (중앙값, 사분위) | 후 |", "|---|---|---|"]
    for nm, k in zip(names, km):
        cell = []
        for key in ("before", "after"):
            v = np.array([r[key][nm] for r in rows if nm in r[key]])
            cell.append(f"{np.median(v):.2f} ({np.percentile(v, 25):.2f}~{np.percentile(v, 75):.2f})" if len(v) else "없음")
        L.append(f"| {k:g} km | {cell[0]} | {cell[1]} |")
    hm = np.array([r["hand_min"] for r in rows])
    L += ["", f"- 전: 규칙이 날던 시간 중앙값 {np.median(hm):.1f}분 ({hm.min():.1f}~{hm.max():.1f}분), 그동안 얻은 POS 평균 "
              f"{np.mean([r['pos_rule'] for r in rows]):.3f}. 후: 첫 결정 시각 모두 {max(r['first_min'] for r in rows):.1f}분.",
          f"- 첫 결정 때 가장 가까운 입자: 전 중앙값 {np.median([r['before']['near_km'] for r in rows]):.1f} km, 후 "
          f"{np.median([r['after']['near_km'] for r in rows]):.1f} km.",
          f"- 관측 모양(후): " + ", ".join(f"{k} {v}" for k, v in net["shapes"].items()),
          f"- 관측 모양(전): " + ", ".join(f"{k} {v}" for k, v in net["old_shapes"].items()),
          f"- 특징 추출 신경망 매개변수: 전 {net['old_params']:,}, 후 {net['params']:,} (정책·가치 머리 제외). 출력 {net['feature']}.",
          f"- 무작위 행동 30걸음: 관측이 공간 정의와 맞음 = {net['ok_space']}, 한 걸음 {net['step_ms']:.0f} ms(맥)."]
    (OUT/"maps7_check.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", nargs="?", default="all", choices=("all", "run", "figure"))
    ap.add_argument("--n", type=int, default=12, help="launches per map")
    a = ap.parse_args()
    if a.what in ("all", "run"):
        run(a.n)
    if a.what in ("all", "figure"):
        figure()


if __name__ == "__main__":
    main()

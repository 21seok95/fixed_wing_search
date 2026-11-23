"""Compute of the multi-UAV joint simulation per 5 s step (user 2026-11-20: does time grow in proportion to the number of suits; sizes the
onboard computer (one Jetson per suit) and informs the model structure). The joint simulation is what every suit would run to predict
its teammates (same weights, shared state) and plan.

- team size: MultiRealEnv (rl_search.rl_multi) on the Jirisan stage-1 case v2j1 #1 (10 km launch) flown by R15 (4.0M, CPU, one thread),
  launches 5 s apart; the first 60 steps (5 min, the transit) are skipped, then 120 steps (10 min over the particles) are timed per
  part: safety filter, flight + camera detection + belief update, belief merge, safety check, observation maps (the 7 maps of R15),
  safety range, teammate tails, extras, and the policy network (all suits in one batch).
- particles: 4 suits with 5000 / 2000 / 1000 planner particles (the case's particles cut to the first n).
- maps: the observation of one suit at the same states with R15's 7 maps and with 5 maps (R13's layout).
- network: one forward pass for batches of 1-10 suits on the CPU (1 and 4 threads) and on the Mac GPU (MPS).
The Jetson figure is an estimate (Mac CPU time x 2.5, range 2-4: an ARM Cortex-A78AE core against an Apple core), until measured on a Jetson.

    python -m rl_search.scripts.multi_design.ma_timing run
    python -m rl_search.scripts.multi_design.ma_timing figure
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import json
import time
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np

OUT = Path("rl_search/data/real/amboy/rl_eval/report")
WORK = OUT/"ma_timing"
MODEL = "rl_search/data/rl_real/r15_chungju/ckpt/model_4000000.zip"
RUN_CFG = "rl_search/data/rl_real/r15_chungju/config.json"
TEAMS = (1, 2, 4, 6, 8, 10)
PARTS = ("filter", "fly_detect", "merge", "safety_check", "obs_maps", "obs_safe_range", "obs_tails", "obs_extras", "policy")
NAMES = dict(filter="안전 필터", fly_detect="비행·카메라 탐지·믿음 갱신", merge="믿음 합치기", safety_check="거리 확인·벌점",
             obs_maps="관측 지도 7장", obs_safe_range="허용 범위", obs_tails="동료 꼬리", obs_extras="동료 벡터", policy="정책 신경망")
WARM, TIMED = 60, 120
JETSON = (2.5, 2.0, 4.0)               # assumed Jetson / Mac CPU time ratio: middle, low, high


def _setup():
    warnings.filterwarnings("ignore")
    import torch
    from stable_baselines3 import PPO
    from rl_search.scripts.rl_eval.signal_check import settings_of
    torch.set_num_threads(1)
    s = settings_of(json.loads(Path(RUN_CFG).read_text()))
    return s, PPO.load(MODEL, device="cpu")


def team_run(k: int, s, model, n_particles: int | None = None) -> dict:
    from rl_search.rl_multi import MultiRealEnv
    from rl_search.rl_real import RealRLConfig
    from rl_search.rl_real_eval import _case
    case, scen, _, _ = _case("v2j1", 3, 0)
    if n_particles:
        scen = replace(scen, track=np.ascontiguousarray(np.asarray(scen.track)[:, :n_particles]),
                       exit_step=np.asarray(scen.exit_step)[:n_particles])
    env = MultiRealEnv(k, RealRLConfig(region="jirisan120", settings=s), seed=0)
    env.reset(options=case.options(scen), gaps=[5.0]*(k - 1))
    tot = {p: 0.0 for p in PARTS}
    for n in range(WARM + TIMED):
        if n == WARM:
            env.timing = {}
        t0 = time.perf_counter()
        obs = [env.observation(i) for i in range(k)]          # the full multi-UAV observation (timed), R15 gets its own part of it
        nvec = 15
        batch = {key: np.stack([o[key][:nvec] if key == "vec" else o[key][:-1] for o in obs]) for key in obs[0]}
        t1 = time.perf_counter()
        acts, _ = model.predict(batch, deterministic=True)
        if n >= WARM:
            tot["policy"] += time.perf_counter() - t1
        env.step([a for a in acts])
        if env.done:
            break
    steps = min(TIMED, n + 1 - WARM)
    for p in PARTS:
        if p != "policy":
            tot[p] = env.timing.get(p, 0.0)
    # step() also builds every suit's observation for its return value: obs_* were timed twice per step (ours + step's) -> halve
    for p in ("obs_maps", "obs_safe_range", "obs_tails", "obs_extras"):
        tot[p] /= 2
    return dict(k=k, particles=int(env.N), steps=steps, ms={p: 1000*tot[p]/steps for p in PARTS},
                found=float(env.found), active=int(env.active.sum()))


def maps_compare(s, n_states: int = 60) -> dict:
    """One suit's observation at the same states with 7 maps (R15) and 5 maps (R13's layout: 1.6 / 6.4 / 25.6 / 51.2 + 102.4 km)."""
    from rl_search.rl_real import RealRLConfig, RealRLEnv
    from rl_search.rl_real_eval import _case
    case, scen, _, _ = _case("v2j1", 3, 0)
    s5 = {key: v for key, v in s.items() if key != "maps7"} | dict(extra_map_m=102400.0, extra_map_lead_s=600.0)
    e7 = RealRLEnv(RealRLConfig(region="jirisan120", settings=dict(s)), seed=0)
    e5 = RealRLEnv(RealRLConfig(region="jirisan120", settings=s5), seed=0)
    for e in (e7, e5):
        e.reset(options=case.options(scen))
    t7 = t5 = 0.0
    for n in range(WARM + n_states):
        e7._advance(np.array([0.3*np.sin(n/7)]))
        if n < WARM:
            continue
        p = e7.env.plant
        e5.env.plant.set_state(p.get_state())
        e5.env.t, e5.env.chi_tgt, e5.env.energy = e7.env.t, e7.env.chi_tgt, e7.env.energy
        e5.env.belief.w[:] = e7.env.belief.w
        a = time.perf_counter(); e7.observation(); b = time.perf_counter(); e5.observation(); c = time.perf_counter()
        t7 += b - a
        t5 += c - b
    return dict(ms7=1000*t7/n_states, ms5=1000*t5/n_states, maps7=len(e7.cfg.maps), maps5=len(e5.cfg.maps))


def net_timing(s, model) -> dict:
    """One forward pass (action) for batches of 1..10 observations: CPU 1 / 4 threads, MPS."""
    import torch
    from rl_search.rl_multi import MultiRealEnv
    from rl_search.rl_real import RealRLConfig
    from rl_search.rl_real_eval import _case
    case, scen, _, _ = _case("v2j1", 3, 0)
    env = MultiRealEnv(1, RealRLConfig(region="jirisan120", settings=s), seed=0)
    env.reset(options=case.options(scen))
    o = env.base_observation(0)
    out = {}
    devs = [("cpu1", "cpu", 1), ("cpu4", "cpu", 4)] + ([("mps", "mps", 1)] if torch.backends.mps.is_available() else [])
    from stable_baselines3 import PPO
    for name, dev, th in devs:
        torch.set_num_threads(th)
        m = PPO.load(MODEL, device=dev)
        res = {}
        for b in (1, 2, 4, 6, 8, 10):
            batch = {key: np.repeat(v[None], b, 0) for key, v in o.items()}
            for _ in range(3):
                m.predict(batch, deterministic=True)
            t0 = time.perf_counter()
            for _ in range(20):
                m.predict(batch, deterministic=True)
            res[b] = 1000*(time.perf_counter() - t0)/20
        out[name] = res
    torch.set_num_threads(1)
    return out


def run():
    s, model = _setup()
    WORK.mkdir(parents=True, exist_ok=True)
    res = dict(teams=[], particles=[], maps=None, net=None)
    for k in TEAMS:
        r = team_run(k, s, model)
        res["teams"].append(r)
        print(f"k={k}: {sum(r['ms'].values()):.1f} ms per step ({r['steps']} steps, {r['active']} airborne) "
              + ", ".join(f"{p} {v:.1f}" for p, v in r["ms"].items()), flush=True)
        (WORK/"timing.json").write_text(json.dumps(res))
    for n in (5000, 2000, 1000):
        r = team_run(4, s, model, n_particles=n)
        res["particles"].append(r)
        print(f"k=4, {n} particles: {sum(r['ms'].values()):.1f} ms per step", flush=True)
    res["maps"] = maps_compare(s)
    print("maps:", res["maps"], flush=True)
    res["net"] = net_timing(s, model)
    print("net:", res["net"], flush=True)
    (WORK/"timing.json").write_text(json.dumps(res))


def figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "AppleGothic"
    plt.rcParams["axes.unicode_minus"] = False
    res = json.loads((WORK/"timing.json").read_text())
    T = res["teams"]
    cols = ["#2a6fdb", "#e07b39", "#3a9a5b", "#8e44ad", "#c0392b", "#16a085", "#7f8c8d", "#d4ac0d", "#34495e"]
    fig = plt.figure(figsize=(21, 11))
    gs = fig.add_gridspec(2, 3, hspace=0.32, wspace=0.25)

    ax = fig.add_subplot(gs[0, 0])                       # (a) per part, by team size
    ks = [r["k"] for r in T]
    bottom = np.zeros(len(T))
    for p, c in zip(PARTS, cols):
        v = np.array([r["ms"][p] for r in T])
        ax.bar(range(len(T)), v, .65, bottom=bottom, color=c, label=NAMES[p])
        bottom += v
    for x, b in enumerate(bottom):
        ax.annotate(f"{b:.0f}", (x, b), textcoords="offset points", xytext=(0, 2), ha="center", fontsize=9)
    ax.set_xticks(range(len(T)))
    ax.set_xticklabels([f"{k}대" for k in ks])
    ax.set_ylabel("공동 시뮬레이션 5초 한 걸음 (ms, 맥 코어 1개)")
    ax.legend(fontsize=7, frameon=False, ncol=2)
    ax.set_title("(가) 대수별 한 걸음 계산 시간과 부분 (입자 위를 훑는 10분 평균)", fontsize=10)

    ax = fig.add_subplot(gs[0, 1])                       # (b) per suit: is it proportional?
    tot = np.array([sum(r["ms"].values()) for r in T])
    ax.plot(ks, tot/np.array(ks), "o-", color="#2a6fdb", label="한 걸음 ÷ 대수 (수트 한 대당)")
    ax.plot(ks, tot[0]*np.array(ks)/np.array(ks), ":", color="#8a8f98", label="그대로 비례할 때")
    for x, y in zip(ks, tot/np.array(ks)):
        ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0, 5), ha="center", fontsize=8)
    ax.set_xlabel("대수")
    ax.set_ylabel("수트 한 대당 ms / 걸음")
    ax.set_ylim(0, max(tot/np.array(ks))*1.4)
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("(나) 대수가 늘 때 한 대당 비용 (낮아지면 비례보다 덜 늘어남)", fontsize=10)

    ax = fig.add_subplot(gs[0, 2])                       # (c) whole joint plans vs transit time
    for minutes, c in ((20, "#3a9a5b"), (60, "#2a6fdb")):
        steps = minutes*12
        mac = tot*steps/1000/60
        ax.plot(ks, mac, "o-", color=c, label=f"{minutes}분 계획: 맥")
        ax.fill_between(ks, mac*JETSON[1], mac*JETSON[2], color=c, alpha=.15)
        ax.plot(ks, mac*JETSON[0], "--", color=c, label=f"{minutes}분 계획: 젯슨 어림(×{JETSON[0]}, 띠 ×{JETSON[1]}~{JETSON[2]})")
    for y, lab in ((5, "10 km 이동 약 5분"), (20, "40 km 이동 약 20분")):
        ax.axhline(y, color="#5b6068", lw=.8, ls=":")
        ax.annotate(lab, (ks[0], y*1.05), fontsize=8, color="#5b6068")
    ax.set_yscale("log")
    ax.set_xlabel("대수")
    ax.set_ylabel("공동 계획 1개 계산 시간 (분, 코어 1개)")
    ax.legend(fontsize=7, frameon=False)
    ax.set_title("(다) 공동 계획 1개를 이동하는 동안 계산할 수 있나", fontsize=10)

    ax = fig.add_subplot(gs[1, 0])                       # (d) particles
    P = res["particles"]
    bottom = np.zeros(len(P))
    for p, c in zip(PARTS, cols):
        v = np.array([r["ms"][p] for r in P])
        ax.bar(range(len(P)), v, .6, bottom=bottom, color=c)
        bottom += v
    for x, b in enumerate(bottom):
        ax.annotate(f"{b:.0f}", (x, b), textcoords="offset points", xytext=(0, 2), ha="center", fontsize=9)
    ax.set_xticks(range(len(P)))
    ax.set_xticklabels([f"입자 {r['particles']:,}" for r in P])
    ax.set_ylabel("4대 한 걸음 (ms)")
    ax.set_title("(라) 계획용 입자 수에 따라 (4대, 색은 (가)와 같음)", fontsize=10)

    ax = fig.add_subplot(gs[1, 1])                       # (e) maps
    M = res["maps"]
    ax.bar([0, 1], [M["ms5"], M["ms7"]], .55, color=["#8a8f98", "#2a6fdb"])
    for x, v in enumerate([M["ms5"], M["ms7"]]):
        ax.annotate(f"{v:.1f} ms", (x, v), textcoords="offset points", xytext=(0, 2), ha="center", fontsize=9)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"지도 {M['maps5']}장 (R13)", f"지도 {M['maps7']}장 (R15)"])
    ax.set_ylabel("수트 한 대 관측 만들기 (ms)")
    ax.set_title("(마) 관측 지도 수에 따라 (같은 상태에서)", fontsize=10)

    ax = fig.add_subplot(gs[1, 2])                       # (f) network
    for (name, lab), c in zip((("cpu1", "CPU 코어 1개"), ("cpu4", "CPU 코어 4개"), ("mps", "맥 GPU (MPS)")), ("#c0392b", "#e07b39", "#2a6fdb")):
        if name in res["net"]:
            bs = sorted(int(b) for b in res["net"][name])
            ax.plot(bs, [res["net"][name][str(b)] for b in bs], "o-", color=c, label=lab)
    ax.set_xlabel("한 번에 계산하는 수트 수")
    ax.set_ylabel("정책 신경망 한 번 (ms)")
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("(바) 정책 신경망: 여러 수트를 한 번에 계산", fontsize=10)
    fig.suptitle("다수 기체 공동 시뮬레이션 계산량 (맥 측정, 지리산 10 km 이륙 사례, R15 정책) · 젯슨은 어림", fontsize=13)
    fig.savefig(OUT/"ma_timing.png", dpi=105, bbox_inches="tight")

    L = ["# 다수 기체 공동 시뮬레이션 계산량 (11-20)", "", "그림 `report/ma_timing.png`. 맥(코어 1개) 측정, 젯슨은 맥 CPU 시간 × 2.5(2~4) 어림.", "",
         "| 대수 | 한 걸음 (ms) | 한 대당 | " + " | ".join(NAMES[p] for p in PARTS) + " |", "|---|---|---|" + "---|"*len(PARTS)]
    for r in T:
        t = sum(r["ms"].values())
        L.append(f"| {r['k']} | {t:.1f} | {t/r['k']:.1f} | " + " | ".join(f"{r['ms'][p]:.1f}" for p in PARTS) + " |")
    L += ["", "| 대수 | 20분 계획 맥 (분) | 젯슨 어림 (분) | 60분 계획 맥 (분) | 젯슨 어림 (분) |", "|---|---|---|---|---|"]
    for r in T:
        t = sum(r["ms"].values())
        L.append(f"| {r['k']} | {t*240/60000:.1f} | {t*240/60000*JETSON[0]:.1f} | {t*720/60000:.1f} | {t*720/60000*JETSON[0]:.1f} |")
    L += ["", "입자 수(4대): " + ", ".join(f"{r['particles']:,}개 {sum(r['ms'].values()):.1f} ms" for r in P),
          f"지도 수(한 대 관측): {M['maps5']}장 {M['ms5']:.1f} ms, {M['maps7']}장 {M['ms7']:.1f} ms",
          "정책 신경망(ms): " + "; ".join(f"{name} " + ", ".join(f"{b}대 {v:.1f}" for b, v in sorted(((int(b), v) for b, v in res['net'][name].items())))
                                    for name in res["net"])]
    (OUT/"ma_timing.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("run", "figure"))
    a = ap.parse_args()
    run() if a.what == "run" else figure()


if __name__ == "__main__":
    main()

"""How long are particles out of view between two looks, and what would other pass gaps change? (replay only)

Takes the saved 1 s flight logs of a chungju_step3 run (paths stay as flown, nothing is re-planned), projects the
particles into the camera at every sample, and for each particle lists its out-of-view gaps. Then the pass rule of
``detection.PassCap`` (pass_mode "scale": each pass finds 0.7 x the model's pass POD; a new pass starts after more
than G seconds out of view) is applied for several G.

Figure: rl_search/data/real/chungju/<run>/pass_gap.png

    python -m rl_search.scripts.real_map.chungju_pass_gap --run step3_flown_tr30 --jobs 5
"""
from __future__ import annotations

import argparse
import json
import warnings
from multiprocessing import get_context

import numpy as np

from rl_search.real_terrain import DATA

GAPS = (2.0, 5.0, 10.0, 15.0, 30.0, 60.0)
POD = 0.7
RADIUS = 4000.0


def one(seed, run, particles=2000):
    warnings.filterwarnings("ignore")
    from rl_search.mission import ParticleBelief
    from rl_search.real_scenario import VehicleScenarios, make_env
    from rl_search.scripts.real_map.chungju_zoom import frames
    env = make_env(seed)
    scen = VehicleScenarios(env.sampler).make(seed, particles)
    o = np.load(DATA/"chungju"/run/f"run_{seed}.npz")
    log = o["log"]
    det = env.detector
    b = ParticleBelief(scen.track, scen.save_dt_s, scen.exit_step, True)
    f, r, u, rate = frames(log)
    seen_t, seen_h = [[] for _ in range(b.n)], [[] for _ in range(b.n)]
    for k, (t, x, y, yaw, phi) in enumerate(log[:, :5]):
        P = b.positions(t)
        near = np.flatnonzero(b.active(t) & (np.abs(P[:, 0] - x) < RADIUS) & (np.abs(P[:, 1] - y) < RADIUS))
        if not len(near):
            continue
        p = det.slice_pod(P[near, 0], P[near, 1], x, y, yaw, phi, rate[k], frame=(f[k][None], r[k][None], u[k][None]))
        for i, q in zip(near[det.last_inside], p[det.last_inside]):
            seen_t[i].append(t)
            seen_h[i].append(-np.log1p(-min(q, 1 - 1e-12)))
    gaps = []
    pos, passes = {}, {}
    viewed = sum(len(t) > 0 for t in seen_t)
    for G in GAPS:
        total, n_pass = 0.0, 0
        for ts, hs in zip(seen_t, seen_h):
            if not ts:
                continue
            ts, hs = np.array(ts), np.array(hs)
            out = np.diff(ts) - 1.0                     # seconds with the particle in no image
            cuts = np.flatnonzero(out > G) + 1
            miss = 1.0
            for h in np.split(hs, cuts):
                miss *= 1 - POD*(-np.expm1(-h.sum()))
            total += 1 - miss
            n_pass += len(cuts) + 1
        pos[G], passes[G] = total/b.n, n_pass/max(viewed, 1)
    for ts in seen_t:
        if len(ts) > 1:
            out = np.diff(ts) - 1.0
            gaps.append(out[out > 0])
    gaps = np.concatenate(gaps) if gaps else np.zeros(0)
    print(f"seed {seed}: replay POS at 2 s {pos[2.0]:.3f} (run {float(o['pos'][-1]):.3f})", flush=True)
    return seed, gaps, pos, passes, float(o["pos"][-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="step3_flown_tr30")
    ap.add_argument("--seeds", type=int, nargs="*", default=list(range(1, 11)))
    ap.add_argument("--jobs", type=int, default=5)
    args = ap.parse_args()
    with get_context("spawn").Pool(args.jobs) as pool:
        res = pool.starmap(one, [(s, args.run) for s in args.seeds])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["AppleGothic", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False
    gaps = np.concatenate([r[1] for r in res])
    pos = np.array([[r[2][G] for G in GAPS] for r in res])
    passes = np.array([[r[3][G] for G in GAPS] for r in res])
    d = pos - pos[:, [0]]

    fig, axs = plt.subplots(1, 3, figsize=(21, 6.8), layout="constrained")
    ax = axs[0]
    bins = np.concatenate([np.arange(.5, 120.5, 1), np.arange(120.5, 1200, 20)])
    ax.hist(gaps, bins=bins, color="#2a4d7a")
    ax.set_xscale("log"); ax.set_yscale("log")
    for g, c, lab in ((2, "#7f8c8d", "2초 (지금)"), (5, "#27ae60", "5초 = RL 결정 1번"), (10, "#e67e22", "10초 = 2번"),
                      (15, "#c0392b", "15초 = 3번"), (30, "#8e44ad", "30초")):
        ax.axvline(g, color=c, lw=1.6, label=lab)
    ax.set_xlabel("화면 밖에 있던 시간 (초, 로그)"); ax.set_ylabel("횟수 (로그)"); ax.legend(fontsize=9)
    ax.set_title(f"같은 particle을 다시 볼 때까지 화면 밖에 있던 시간 ({len(gaps):,}번)", fontsize=11)
    ax = axs[1]
    xs = np.sort(gaps)
    ax.plot(xs, np.arange(1, len(xs) + 1)/len(xs), color="#2a4d7a", lw=2.2)
    ax.set_xscale("log"); ax.set_ylim(0, 1)
    for g in (2, 5, 10, 15, 30, 60):
        v = float((gaps <= g).mean())
        ax.plot([g], [v], "o", color="#c0392b")
        ax.annotate(f"{g}초: {v:.0%}", (g, v), textcoords="offset points", xytext=(6, -12), fontsize=9)
    ax.set_xlabel("화면 밖에 있던 시간 (초, 로그)"); ax.set_ylabel("이 시간 이하인 비율")
    ax.set_title("누적 분포", fontsize=11)
    ax = axs[2]
    x = np.arange(len(GAPS))
    ax.bar(x, d.mean(0), .6, yerr=d.std(0, ddof=1)/np.sqrt(len(res)), color="#2a4d7a", capsize=4)
    ax.axhline(0, color="k", lw=.8)
    ax.set_xticks(x); ax.set_xticklabels([f"{g:g}초" for g in GAPS]); ax.set_xlabel("새 통과로 치는 간격")
    ax.set_ylabel("POS 차이 (2초 대비)")
    ax2 = ax.twinx()
    ax2.plot(x, passes.mean(0), "o-", color="#c0392b", lw=2)
    ax2.set_ylabel("화면에 든 particle 하나당 통과 횟수", color="#c0392b")
    ax.set_title("같은 경로에서 간격만 바꿨을 때 (막대 = POS 차이, 선 = 통과 횟수)", fontsize=11)
    fig.suptitle(f"통과 간격 재계산: 경로는 그대로, 탐지만 다시 계산 ({args.run}, 시나리오 {len(res)}개, 통과마다 0.7 × 모델 POD)", fontsize=13)
    out = DATA/"chungju"/args.run/"pass_gap.png"
    fig.savefig(out, dpi=95)
    summary = dict(gaps=list(GAPS), pos=pos.mean(0).tolist(), diff=d.mean(0).tolist(),
                   diff_se=(d.std(0, ddof=1)/np.sqrt(len(res))).tolist(), passes=passes.mean(0).tolist(),
                   share_le={str(g): float((gaps <= g).mean()) for g in (2, 5, 10, 15, 30, 60)},
                   replay_vs_run=[(r[0], r[2][2.0], r[4]) for r in res])
    (out.with_suffix(".json")).write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "replay_vs_run"}, indent=0))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()

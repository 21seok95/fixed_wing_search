"""Multi-UAV step 1 (docs/multi_agent_design.md "구현 준비 (2026-11-19)"): the multi-UAV env (rl_search.rl_multi) flown by R15 on every UAV
(no teammate inputs: R15 sees the shared belief only), before any multi-UAV training. The three Jirisan validation cases (v2j) with 1-4 UAVs
(safety filter on) and 3 UAVs without the filter. Shows the problem to solve and answers the user's question "can a teammate's next moves
be read from its tail?" (straight extrapolation of the tail vs where the UAV really is 30 s - 10 min later).

    python -m rl_search.scripts.multi_design.ma_step1 run
    python -m rl_search.scripts.multi_design.ma_step1 figure
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np

OUT = Path("rl_search/data/real/amboy/rl_eval/report")
WORK = OUT/"ma_step1"
MODEL = "rl_search/data/rl_real/r15_chungju/ckpt/model_4000000.zip"
RUN_CFG = "rl_search/data/rl_real/r15_chungju/config.json"
COLORS = ("#2a6fdb", "#e07b39", "#3a9a5b", "#8e44ad")     # UAV 1-4
HORIZONS = (30, 60, 120, 300, 600)


def fly(k: int, case_k: int, safety: bool, model, s, keep_obs_at: float | None = None) -> dict:
    from rl_search.rl_multi import MultiRealEnv
    from rl_search.rl_real import RealRLConfig
    from rl_search.rl_real_eval import _case
    case, scen, _, _ = _case("v2j", 3, case_k)
    env = MultiRealEnv(k, RealRLConfig(region="jirisan120", settings=s), seed=case_k, safety=safety)
    gaps = list(np.random.default_rng(1000 + case_k).choice((5.0, 10.0, 15.0), 3))[:k - 1]
    env.reset(options=case.options(scen), gaps=gaps)
    snap = None
    t0 = time.perf_counter()
    while not env.done:
        obs = [env.base_observation(i) for i in range(k)]
        batch = {key: np.stack([o[key] for o in obs]) for key in obs[0]}
        acts, _ = model.predict(batch, deterministic=True)
        if keep_obs_at is not None and snap is None and env.T >= keep_obs_at:
            full = env.observation(1)
            snap = {key: v.tolist() for key, v in full.items() if key in ("m6400", "m25600", "m102400", "vec")}
            snap["pose"] = env.uavs[1].env.plant.pose.tolist()
        env.step([a for a in acts])
    lg = env.log
    return dict(k=k, case=case_k, safety=safety, launch_t=env.launch_t.tolist(), t=lg["t"], found=lg["found"], dist=lg["dist"],
                cut=np.array(lg["cut"]).tolist(), viol=np.array(lg["viol"]).tolist(), share=np.array(lg["share"]).tolist(),
                seen=np.array(lg["seen"]).tolist(), hist=[np.array(h).tolist() for h in env.hist], lkp=case.lkp, start=case.start,
                sec=time.perf_counter() - t0, snap=snap)


def run():
    warnings.filterwarnings("ignore")
    import torch
    from stable_baselines3 import PPO
    from rl_search.scripts.rl_eval.signal_check import settings_of
    torch.set_num_threads(2)
    s = settings_of(json.loads(Path(RUN_CFG).read_text()))
    model = PPO.load(MODEL, device="cpu")
    WORK.mkdir(parents=True, exist_ok=True)
    rows = []
    for case_k in range(3):
        for k, safety in ((1, True), (2, True), (3, True), (4, True), (3, False)):
            r = fly(k, case_k, safety, model, s, keep_obs_at=900.0 if (case_k == 0 and k == 3 and safety) else None)
            rows.append(r)
            print(f"case {case_k + 1} k={k} filter={'on' if safety else 'off'}: team POS {r['found'][-1]:.3f}, min distance {min(r['dist']):.0f} m, "
                  f"cuts {int((np.array(r['cut']) > 0).sum())}, violations {int(np.array(r['viol']).sum())}, {r['sec']:.0f} s", flush=True)
            (WORK/"rows.json").write_text(json.dumps(rows))


def predictability(rows) -> dict:
    """Straight extrapolation of a UAV's tail (position + heading from its last 1 s move, 33 m/s) vs its real position after each horizon:
    error (m) and the angle between its heading and the bearing to where it really is."""
    err = {h: [] for h in HORIZONS}
    ang = {h: [] for h in HORIZONS}
    for r in rows:
        if not r["safety"]:
            continue
        for h in r["hist"]:
            a = np.array(h)
            if len(a) < 200:
                continue
            t = a[:, 0]
            for i in range(120, len(a), 30):                    # from 2 min after the launch, every 30 s
                if t[i] - t[i - 1] <= 0:
                    continue
                d = a[i, 1:] - a[i - 1, 1:]
                v = np.hypot(*d)/(t[i] - t[i - 1])
                if v < 1:
                    continue
                hd = np.arctan2(d[1], d[0])
                for hz in HORIZONS:
                    j = np.searchsorted(t, t[i] + hz)
                    if j >= len(a):
                        continue
                    pred = a[i, 1:] + 33.0*hz*np.array([np.cos(hd), np.sin(hd)])
                    err[hz].append(float(np.hypot(*(pred - a[j, 1:]))))
                    b = np.arctan2(a[j, 2] - a[i, 2], a[j, 1] - a[i, 1])
                    ang[hz].append(float(np.degrees(abs((b - hd + np.pi) % (2*np.pi) - np.pi))))
    return dict(err=err, ang=ang)


def figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    plt.rcParams["font.family"] = "AppleGothic"
    plt.rcParams["axes.unicode_minus"] = False
    rows = json.loads((WORK/"rows.json").read_text())
    on = [r for r in rows if r["safety"]]
    fig = plt.figure(figsize=(25, 12))
    gs = fig.add_gridspec(2, 4, hspace=0.34, wspace=0.26)

    # (a) paths of case 1, 3 UAVs
    ax = fig.add_subplot(gs[0, 0])
    bgp = OUT/"launch_ring_check"/"jirisan_bg.npz"
    r3 = next(r for r in on if r["k"] == 3 and r["case"] == 0)
    if bgp.exists():
        b = np.load(bgp)
        ax_ = b["axis"]/1000
        ax.imshow(b["closed"].astype(np.uint8), origin="lower", extent=(ax_[0], ax_[-1], ax_[0], ax_[-1]),
                  cmap=ListedColormap(["#ffffff", "#e6e8eb"]), vmin=0, vmax=1, interpolation="nearest")
    for i, h in enumerate(r3["hist"]):
        a = np.array(h)/np.array([1, 1000, 1000])
        ax.plot(a[:, 1], a[:, 2], color=COLORS[i], lw=1.0, label=f"{i + 1}번 (이륙 {r3['launch_t'][i]:.0f}초)")
    ax.plot(*np.array(r3["lkp"])/1000, "*", color="#d9480f", ms=13, label="LKP")
    ax.plot(*np.array(r3["start"][:2])/1000, "^", color="#222", ms=9, label="기지")
    allp = np.concatenate([np.array(h)[:, 1:] for h in r3["hist"]])/1000
    c = (allp.min(0) + allp.max(0))/2
    half = max(float((allp.max(0) - allp.min(0)).max())/2 + 2, 8)
    ax.set_xlim(c[0] - half, c[0] + half)
    ax.set_ylim(c[1] - half, c[1] + half)
    ax.set_aspect("equal")
    ax.set_xlabel("동쪽 (km)")
    ax.set_ylabel("북쪽 (km)")
    ax.legend(fontsize=8, frameon=False)
    ax.set_title(f"(가) 지리산 사례 1, 3대 모두 R15(동료 정보 없음, 믿음만 공유)\n팀 POS {r3['found'][-1]:.3f}", fontsize=10)

    # (b) team POS over time by team size
    ax = fig.add_subplot(gs[0, 1])
    for k, col in zip((1, 2, 3, 4), ("#8a8f98", COLORS[0], COLORS[1], COLORS[2])):
        rr = [r for r in on if r["k"] == k]
        n = min(len(r["found"]) for r in rr)
        t = np.array(rr[0]["t"][:n])/60
        f = np.mean([r["found"][:n] for r in rr], axis=0)
        ax.plot(t, f, color=col, lw=1.6, label=f"{k}대: 60분 {f[-1]:.3f}")
    ax.set_xlabel("이륙 뒤 (분)")
    ax.set_ylabel("팀 POS (계획용 입자, 사례 3개 평균)")
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("(나) 대수별 팀 POS: R15를 각자 쓰면 대수를 늘려도 덜 오름", fontsize=10)

    # (c) per case, per team size
    ax = fig.add_subplot(gs[0, 2])
    w = .2
    for j, k in enumerate((1, 2, 3, 4)):
        v = [next(r for r in on if r["k"] == k and r["case"] == c_)["found"][-1] for c_ in range(3)]
        x = np.arange(3) + (j - 1.5)*w
        ax.bar(x, v, w*.9, color=("#8a8f98", COLORS[0], COLORS[1], COLORS[2])[j], label=f"{k}대")
        for xi, vi in zip(x, v):
            ax.annotate(f"{vi:.2f}", (xi, vi), textcoords="offset points", xytext=(0, 2), ha="center", fontsize=7)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["사례 1", "사례 2", "사례 3"])
    ax.set_ylabel("60분 팀 POS")
    ax.legend(fontsize=8, frameon=False, ncol=4)
    ax.set_title("(다) 사례별: 한 대 더할 때 얻는 것", fontsize=10)

    # (d) separation, case 1, 3 UAVs, filter on / off
    ax = fig.add_subplot(gs[0, 3])
    r3off = next(r for r in rows if r["k"] == 3 and r["case"] == 0 and not r["safety"])
    for r, col, lab in ((r3off, "#d9480f", "필터 없음"), (r3, COLORS[0], "필터 있음")):
        d = np.array(r["dist"], float)
        ax.plot(np.array(r["t"])/60, np.where(np.isfinite(d), d, np.nan)/1000, color=col, lw=1, label=f"{lab}: 최소 {np.nanmin(d):.0f} m")
    for y, lab in ((0.15, "150 m 금지"), (0.2, "200 m 벌점")):
        ax.axhline(y, color="#5b6068", lw=.8, ls=":")
        ax.annotate(lab, (0.5, y), fontsize=8, color="#5b6068")
    ax.set_yscale("log")
    ax.set_xlabel("이륙 뒤 (분)")
    ax.set_ylabel("가장 가까운 두 기체 사이 (km, 5초마다 최소)")
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("(라) 사례 1, 3대: 기체 사이 거리", fontsize=10)

    # (e) concurrent overlap and own share
    ax = fig.add_subplot(gs[1, 0])
    for j, k in enumerate((2, 3, 4)):
        rr = [r for r in on if r["k"] == k]
        seen = np.array([np.sum(r["seen"], axis=0).sum() for r in rr])
        team = np.array([r["found"][-1] for r in rr])
        ov = (seen - team)/np.maximum(seen, 1e-9)
        ax.bar(j, ov.mean()*100, .6, color=COLORS[j])
        ax.annotate(f"{ov.mean()*100:.1f}%", (j, ov.mean()*100), textcoords="offset points", xytext=(0, 2), ha="center", fontsize=9)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["2대", "3대", "4대"])
    ax.set_ylabel("같은 5초에 겹쳐 본 몫 (각자 본 것 합 중 %)")
    ax.set_title("(마) 동시에 같은 입자를 본 몫 (자기 몫과 '각자 찾은 것'의 차이)", fontsize=10)

    # (f) how close the UAVs fly
    ax = fig.add_subplot(gs[1, 1])
    for j, k in enumerate((2, 3, 4)):
        rr = [r for r in on if r["k"] == k]
        fr = []
        for lim in (1000, 2000, 5000):
            vals = []
            for r in rr:
                d = np.array(r["dist"], float)
                d = d[np.isfinite(d)]
                vals.append(float((d < lim).mean()) if len(d) else 0.0)
            fr.append(np.mean(vals)*100)
        ax.plot([1, 2, 5], fr, "o-", color=COLORS[j], label=f"{k}대")
    ax.set_xlabel("가까운 두 기체 사이가 이 거리 안 (km)")
    ax.set_ylabel("비행 시간 중 %")
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("(바) 얼마나 붙어 나나 (같은 덩어리를 같이 훑는 정도)", fontsize=10)

    # (g) predictability of a teammate from its tail
    ax = fig.add_subplot(gs[1, 2])
    pr = predictability(rows)
    hz = [h for h in HORIZONS if pr["err"][h]]
    med = [np.median(pr["err"][h])/1000 for h in hz]
    p90 = [np.quantile(pr["err"][h], .9)/1000 for h in hz]
    ax.plot(np.array(hz)/60, med, "o-", color=COLORS[0], label="꼬리를 곧게 늘인 예측의 오차: 중앙값")
    ax.plot(np.array(hz)/60, p90, "o--", color=COLORS[0], alpha=.6, label="90%")
    ax.plot(np.array(hz)/60, 33*np.array(hz)/1000, color="#8a8f98", lw=.8, ls=":", label="그동안 나는 거리")
    for half, lab in ((0.8, "1.6 km 지도 반폭"), (3.2, "6.4 km 지도"), (12.8, "25.6 km 지도")):
        ax.axhline(half, color="#c9cdd2", lw=.8)
        ax.annotate(lab, (0.4, half*1.04), fontsize=7, color="#5b6068")
    ax.set_yscale("log")
    ax.set_xlabel("몇 분 뒤 위치를 맞히나")
    ax.set_ylabel("오차 (km)")
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax.set_title("(사) 동료 꼬리로 앞날을 맞힐 수 있나 (R15 비행 전부)", fontsize=10)

    # (h) heading vs where it really goes
    ax = fig.add_subplot(gs[1, 3])
    for j, h in enumerate(hz):
        a = np.array(pr["ang"][h])
        ax.hist(a, bins=np.arange(0, 181, 10), histtype="step", lw=1.4, color=plt.cm.viridis(j/max(len(hz) - 1, 1)),
                label=f"{h/60:g}분 뒤: 45° 안 {np.mean(a < 45)*100:.0f}%", density=True)
    ax.set_xlabel("지금 기수와 실제 간 쪽의 각도 차 (°)")
    ax.set_ylabel("밀도")
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("(아) 지금 향하는 쪽이 몇 분 뒤 있는 쪽인가", fontsize=10)
    fig.suptitle("다수 기체 1단계: 다수 기체 환경을 R15로 날린 기준선(학습 전) · 지리산 검증 3개 사례 · 같은 기지, 5~15초 간격 이륙, 믿음 공유, 안전 필터 150 m",
                 fontsize=13)
    fig.savefig(OUT/"ma_step1.png", dpi=105, bbox_inches="tight")

    L = ["# 다수 기체 1단계 확인 (11-19)", "", "그림 `report/ma_step1.png`. R15 400만(동료 입력 없음)을 모든 기체가 씀, 팀 POS는 계획용 입자(채점용 입자 재생은 아직).", "",
         "| 사례 | 대수 | 필터 | 60분 팀 POS | 가장 가까운 거리 (m) | 필터가 자른 결정 | 150 m 안 위반 | 겹쳐 본 몫 |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        seen = np.sum(r["seen"], axis=0).sum()
        ov = (seen - r["found"][-1])/seen if seen > 0 else 0.0
        d = np.array(r["dist"], float)
        dm = np.nanmin(np.where(np.isfinite(d), d, np.nan)) if np.isfinite(d).any() else float("nan")
        L.append(f"| {r['case'] + 1} | {r['k']} | {'있음' if r['safety'] else '없음'} | {r['found'][-1]:.3f} | {dm:.0f} | "
                 f"{int((np.array(r['cut']) > 0).sum())} | {int(np.array(r['viol']).sum())} | {ov*100:.1f}% |")
    L += ["", "| 몇 분 뒤 | 곧게 늘인 예측 오차 중앙값 (km) | 90% (km) | 기수와 실제 방향 45° 안 |", "|---|---|---|---|"]
    for h in hz:
        L.append(f"| {h/60:g} | {np.median(pr['err'][h])/1000:.2f} | {np.quantile(pr['err'][h], .9)/1000:.2f} | {np.mean(np.array(pr['ang'][h]) < 45)*100:.0f}% |")
    (OUT/"ma_step1.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("run", "figure"))
    a = ap.parse_args()
    run() if a.what == "run" else figure()


if __name__ == "__main__":
    main()

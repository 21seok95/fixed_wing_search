"""Is the circling tied to time (the energy-left input) or to the state? (user 2026-11-17, after the Chungju web flights of R14 road:
small circles appear also where the particles are not concentrated.)

The user's Chungju web flights are flown again with the same model, scenario and settings (rl_search.scripts.real_map.web_rl), the flight
itself unchanged (60 min, same particles), only the energy-left entry of the observation vector changed:
  base - as it is (must give the web run's POS)
  slow - energy shown as if it were used at half the rate: 1 - 0.5 x (used share)
  full - energy shown as full (1) all flight
Per decision: the UAV pose, POS, the energy shown, circling (net heading change of 330 deg or more within 3 min, midstart_check.circling),
and the state around the UAV: remaining probability within 1 km and 3 km, the heaviest 1 km cell of the remaining probability and how
far it is.

    python -m rl_search.scripts.rl_eval.circling_cause run --variant base     (and slow, full; one process each)
    python -m rl_search.scripts.rl_eval.circling_cause figure
"""
from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np

RUNS = Path("rl_search/data/real/chungju/web_runs")
OUT = Path("rl_search/data/real/amboy/rl_eval/report")
WORK = OUT/"circling_cause"
VARIANTS = ("base", "slow", "full")
VN = {"base": "원래", "slow": "에너지가 절반 속도로 줄게 보임", "full": "에너지가 늘 가득으로 보임"}
VC = {"base": "#1f1f1e", "slow": "#2a78d6", "full": "#eb6834"}
E_IDX = 3                                                   # rl_real.VEC_NAMES: energy_left


def web_runs(day="20261117"):
    out = []
    for d in sorted(RUNS.glob(f"{day}-*")):
        m = json.loads((d/"meta.json").read_text())
        if (m.get("view") or {}).get("region") == "chungju" and m.get("model"):
            out.append(m)
    return out


def build(m):
    """Scenario of a web RL run, made as web_rl._run makes it (no hold-out)."""
    from rl_search.real_scenario import VehicleScenarios
    from rl_search.rl_eval_set import Case
    from rl_search.rl_real import preload
    from rl_search.vehicle import VehicleConfig
    view = m["view"]
    launch_s = 60.0*float(m.get("launch_min", 0.0) or 0.0)
    st = preload("chungju")
    vc = VehicleConfig(**m["config"])
    vc = replace(vc, duration_s=max(float(vc.duration_s), launch_s + 3600.0 + 600.0))
    vs = VehicleScenarios(st["sampler"], vc, region="chungju", net=st["net"])
    seed, heading = int(view["seed"]), float(view["heading"])
    scen = vs.make_at(seed, view["lkp"], heading, view.get("start"))
    case = Case("web", 3, 0, seed, [float(v) for v in scen.lkp], heading, 0.0,
                [float(scen.start[0]), float(scen.start[1]), float(np.radians(scen.start_heading_deg))], {"web": True}, t_launch_s=launch_s,
                region="chungju")
    return case, scen


def local_state(env):
    e = env.env
    b = e.belief
    P = b.positions(e.t)
    p = b.w/b.n
    x, y = e.plant.pose[:2]
    d = np.hypot(P[:, 0] - x, P[:, 1] - y)
    q = np.floor(P/1000.0).astype(int)
    keys, inv = np.unique(q, axis=0, return_inverse=True)
    mass = np.bincount(inv.ravel(), weights=p)
    j = int(np.argmax(mass))
    c = (keys[j] + 0.5)*1000.0
    return float(p[d <= 1000].sum()), float(p[d <= 3000].sum()), float(mass[j]), float(np.hypot(c[0] - x, c[1] - y)), float(b.mass)


def run(variant: str):
    warnings.filterwarnings("ignore")
    from rl_search.rl_real_eval import _env, _model
    from rl_search.scripts.real_map.web_rl import run_dir
    from rl_search.scripts.rl_eval.signal_check import settings_of
    WORK.mkdir(parents=True, exist_ok=True)
    out = []
    for i, m in enumerate(web_runs()):
        model = _model(m["model"])
        settings = settings_of(json.loads((run_dir(Path(m["model"]))/"config.json").read_text()))
        case, scen = build(m)
        env = _env(settings, "chungju")
        orig = type(env).observation

        def observation(self=env):
            o = orig(self)
            e_true = float(o["vec"][E_IDX])
            if variant == "slow":
                o["vec"][E_IDX] = 1.0 - 0.5*(1.0 - e_true)
            elif variant == "full":
                o["vec"][E_IDX] = 1.0
            return o
        env.observation = observation
        obs, info = env.reset(options=case.options(scen))
        rec = []
        term = trunc = False
        while not (term or trunc):
            a, _ = model.predict(obs, deterministic=True)
            e_shown = float(obs["vec"][E_IDX])
            obs, r, term, trunc, info = env.step(a)
            x, y, yaw = env.env.plant.pose
            rec.append((env.env.t, x, y, yaw, float(info["POS"]), e_shown, float(np.ravel(a)[0]), *local_state(env)))
        del env.observation
        rec = np.array(rec)
        out.append(dict(i=i + 1, id=m["id"], web_final=m["final"], final=float(rec[-1, 4]), rec=rec.tolist()))
        print(f"{variant} {i + 1}: POS {rec[-1, 4]:.3f} (web {m['final']:.3f})", flush=True)
    (WORK/f"{variant}.json").write_text(json.dumps(out))


def circling(yaw, dt=5.0, half_window_s=90.0):
    u = np.unwrap(np.asarray(yaw, float))
    k = int(half_window_s/dt)
    out = np.zeros(len(u), bool)
    for i in range(len(u)):
        a, b = max(0, i - k), min(len(u) - 1, i + k)
        out[i] = abs(u[b] - u[a]) >= np.radians(330)
    return out


def figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["AppleGothic", "NanumGothic", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False
    INK, MUTED, GRID, BG = "#1f1f1e", "#6b6b67", "#e4e3de", "#fcfcfb"
    D = {v: json.loads((WORK/f"{v}.json").read_text()) for v in VARIANTS if (WORK/f"{v}.json").exists()}
    fig = plt.figure(figsize=(24, 15), facecolor=BG)
    gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 1.1])
    bins = np.arange(0, 61, 5)
    # 1. circling share by clock time, per variant
    ax = fig.add_subplot(gs[0, 0:2])
    for v, runs in D.items():
        t_all, c_all = [], []
        for r in runs:
            a = np.array(r["rec"])
            t_all.append(a[:, 0]/60); c_all.append(circling(a[:, 3]))
        t, c = np.concatenate(t_all), np.concatenate(c_all)
        k = np.digitize(t, bins) - 1
        sh = [100*c[k == j].mean() if (k == j).any() else np.nan for j in range(len(bins) - 1)]
        ax.plot(bins[:-1] + 2.5, sh, color=VC[v], lw=2.2, marker="o", label=f"{VN[v]} (전체 {100*c.mean():.0f}%)")
    ax.set_xlabel("신고 뒤 분 (시계)", fontsize=9, color=MUTED); ax.set_ylabel("원 선회 중인 결정의 비율 (%)", fontsize=9, color=MUTED)
    ax.set_title("① 시계 기준: 5분마다 원 선회 비율 (비행 15개 합)", loc="left", fontsize=11, color=INK)
    ax.legend(fontsize=9, frameon=False)
    # 2. circling share by energy shown
    ax = fig.add_subplot(gs[0, 2:4])
    eb = np.linspace(0, 1, 11)
    for v, runs in D.items():
        e_all, c_all = [], []
        for r in runs:
            a = np.array(r["rec"])
            e_all.append(a[:, 5]); c_all.append(circling(a[:, 3]))
        e, c = np.concatenate(e_all), np.concatenate(c_all)
        k = np.clip(np.digitize(e, eb) - 1, 0, len(eb) - 2)
        sh = [100*c[k == j].mean() if (k == j).sum() > 20 else np.nan for j in range(len(eb) - 1)]
        ax.plot(eb[:-1] + 0.05, sh, color=VC[v], lw=2.2, marker="o", label=VN[v])
    ax.invert_xaxis()
    ax.set_xlabel("정책에게 보인 남은 에너지 (1 = 가득)", fontsize=9, color=MUTED); ax.set_ylabel("원 선회 비율 (%)", fontsize=9, color=MUTED)
    ax.set_title("② 보인 에너지 기준 원 선회 비율 (시간 때문이면 세 선이 같은 에너지에서 겹침)", loc="left", fontsize=11, color=INK)
    ax.legend(fontsize=9, frameon=False)
    # 3. per flight: POS and circling share
    ax = fig.add_subplot(gs[1, 0:2])
    n = len(next(iter(D.values())))
    w = 0.27
    for j, (v, runs) in enumerate(D.items()):
        ax.bar(np.arange(n) + (j - 1)*w, [100*circling(np.array(r["rec"])[:, 3]).mean() for r in runs], width=w, color=VC[v], label=VN[v])
    ax.set_xticks(range(n), [str(r["i"]) for r in next(iter(D.values()))])
    ax.set_xlabel("비행 번호 (앞 그림과 같음)", fontsize=9, color=MUTED); ax.set_ylabel("원 선회 비율 (%)", fontsize=9, color=MUTED)
    ax.set_title("③ 비행마다 원 선회 비율", loc="left", fontsize=11, color=INK)
    ax.legend(fontsize=8.5, frameon=False)
    ax = fig.add_subplot(gs[1, 2:4])
    for j, (v, runs) in enumerate(D.items()):
        ax.bar(np.arange(n) + (j - 1)*w, [r["final"] for r in runs], width=w, color=VC[v], label=VN[v])
    if "base" in D:
        ax.scatter(np.arange(n) - w, [r["web_final"] for r in D["base"]], marker="_", s=300, color="#d6336c", zorder=5, label="웹 결과")
    ax.set_xticks(range(n), [str(r["i"]) for r in next(iter(D.values()))])
    ax.set_ylabel("찾은 POS", fontsize=9, color=MUTED)
    ax.set_title("④ 비행마다 찾은 POS (빨간 줄 = 웹에서 나온 값: '원래'와 같아야 함)", loc="left", fontsize=11, color=INK)
    ax.legend(fontsize=8.5, frameon=False)
    # 4. state when circling vs not (base)
    if "base" in D:
        a = np.concatenate([np.column_stack([np.array(r["rec"]), circling(np.array(r["rec"])[:, 3])]) for r in D["base"]])
        circ = a[:, -1] > 0
        items = [(7, "기체 1 km 안 남은 확률", 1e-5), (8, "기체 3 km 안 남은 확률", 1e-5), (9, "가장 무거운 1 km 칸의 남은 확률", 1e-5),
                 (10, "가장 무거운 칸까지 거리 (km)", None)]
        for col, (k, ttl, floor) in enumerate(items):
            ax = fig.add_subplot(gs[2, col])
            vals = a[:, k]/(1000 if floor is None else 1)
            if floor is not None:
                vals = np.log10(np.maximum(vals, floor))
            lo, hi = np.percentile(vals, 1), np.percentile(vals, 99)
            bb = np.linspace(lo, hi, 30)
            ax.hist(vals[~circ], bins=bb, density=True, color="#9a9a95", alpha=0.7, label=f"원 선회 아님 ({(~circ).sum()}결정)")
            ax.hist(vals[circ], bins=bb, density=True, color="#d6336c", alpha=0.6, label=f"원 선회 중 ({circ.sum()}결정)")
            ax.set_title(f"⑤ '원래' 비행: {ttl}" + (" (log10)" if floor is not None else ""), loc="left", fontsize=10, color=INK)
            ax.legend(fontsize=8, frameon=False)
    for a_ in fig.axes:
        a_.set_facecolor(BG); a_.tick_params(colors=MUTED, labelsize=8.5); a_.grid(color=GRID, lw=.5)
        for s in a_.spines.values():
            s.set_color(GRID)
    fig.suptitle("원 선회는 시간(남은 에너지 입력) 때문인가, 상태 때문인가: R14 도로형, 웹 충주 비행 15개를 비행은 그대로 두고 정책에게 보이는 에너지만 바꿔 다시 날림",
                 x=0.01, ha="left", fontsize=13, color=INK)
    fig.tight_layout()
    fig.savefig(OUT/"circling_cause.png", dpi=80, facecolor=BG)
    for v, runs in D.items():
        c = np.concatenate([circling(np.array(r["rec"])[:, 3]) for r in runs])
        print(v, "circling", round(100*c.mean(), 1), "% ; POS mean", round(float(np.mean([r["final"] for r in runs])), 3))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=("run", "figure"))
    ap.add_argument("--variant", choices=VARIANTS, default="base")
    a = ap.parse_args()
    if a.what == "run":
        run(a.variant)
    else:
        figure()


if __name__ == "__main__":
    main()

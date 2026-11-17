"""What a mid-flight start on an already-searched belief looks like (user 2026-11-17: 40 % of training episodes would start this way), shown
with an existing model on validation cases, before building it into training.

A case is flown from its launch with the model up to T minutes after the launch: that flight searched the belief (the "snapshot", the same
state a training flight would save). From that state the aircraft is put at one of four start points and the model flies the rest of the
flight with the energy that is left:
  - kept:    where the flight was (the normal continuation)
  - inside:  0-1 km from a particle drawn by its remaining weight
  - edge:    3-6 km outside the remaining belief (no remaining particle within 3 km)
  - far:     10-15 km from the nearest remaining particle
The env is deterministic, so replaying the first flight's actions rebuilds the state at T exactly for every start point.

    python -m rl_search.scripts.rl_eval.midstart_check --model rl_search/data/rl_real/r12_2map/ckpt/model_6000000.zip --label r12_2map-6000000
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np

VARIANTS = ("kept", "inside", "edge", "far")
VN = {"kept": "그대로", "inside": "남은 확률 안", "edge": "가장자리 바깥 3~6 km", "far": "먼 곳 10~15 km"}
VC = {"kept": "#1f1f1e", "inside": "#2a78d6", "edge": "#eb6834", "far": "#d6336c"}
CASES = (("bval", 0), ("bval", 12), ("nval", 1), ("nval", 8))      # Barstow road 1, 13 (R12 circled late), Amboy mixed 2, Amboy desert 9


def start_point(kind, P, w, half, rng, pose):
    """(x, y, heading) of a start point of this kind, from the remaining particles P (positions now) with weights w."""
    if kind == "kept":
        return pose
    for _ in range(4000):
        i = rng.choice(len(w), p=w/w.sum())
        if kind == "inside":
            r, a = rng.uniform(0, 1000), rng.uniform(-np.pi, np.pi)
        elif kind == "edge":
            r, a = rng.uniform(3000, 6000), rng.uniform(-np.pi, np.pi)
        else:
            r, a = rng.uniform(10000, 15000), rng.uniform(-np.pi, np.pi)
        x, y = P[i, 0] + r*np.cos(a), P[i, 1] + r*np.sin(a)
        if max(abs(x), abs(y)) > half - 2000:
            continue
        d = np.hypot(P[:, 0] - x, P[:, 1] - y).min()
        if kind == "edge" and not (3000 <= d <= 6000):
            continue
        if kind == "far" and not (10000 <= d <= 15000):
            continue
        return np.array([x, y, rng.uniform(-np.pi, np.pi)])
    return None


def circling(yaw, dt=5.0, half_window_s=90.0):
    """Per decision: net heading change over the 3 min around it is 330 deg or more (one way)."""
    u = np.unwrap(np.asarray(yaw, float))
    k = int(half_window_s/dt)
    out = np.zeros(len(u), bool)
    for i in range(len(u)):
        a, b = max(0, i - k), min(len(u) - 1, i + k)
        out[i] = abs(u[b] - u[a]) >= np.radians(330)
    return out


def main():
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--minutes", nargs="+", type=float, default=[25.0, 40.0], help="snapshot times after the launch")
    ap.add_argument("--out", type=Path, default=Path("rl_search/data/real/amboy/rl_eval/report"))
    args = ap.parse_args()
    from rl_search.rl_eval_set import case_type
    from rl_search.rl_real_eval import _case, _env, _model
    from rl_search.scripts.rl_eval.signal_check import settings_of
    mp = Path(args.model)
    run = mp.parent.parent if mp.parent.name == "ckpt" else mp.parent
    settings = settings_of(json.loads((run/"config.json").read_text()))
    model = _model(args.model)
    rows, panels = [], []
    for sp, k in CASES:
        case, scen, vres, hold = _case(sp, 3, k)
        env = _env(settings, case.region)
        e = env.env
        t_l = float(case.t_launch_s)
        # the first flight: actions and the state at each snapshot time
        obs, info = env.reset(options=case.options(scen))
        acts, term, trunc = [], False, False
        snaps = {}
        while not (term or trunc):
            m_now = (e.t - t_l)/60.0
            for T in args.minutes:
                if T not in snaps and m_now >= T:
                    snaps[T] = (len(acts), e.t, e.plant.pose.copy(), float(info["POS"]), e.belief.positions(e.t).copy(), e.belief.w.copy())
            a, _ = model.predict(obs, deterministic=True)
            acts.append(float(np.ravel(a)[0]))
            obs, r, term, trunc, info = env.step(a)
        for T, (n_pre, t_s, pose, pos0, P, w) in snaps.items():
            live = w > 0
            rng = np.random.default_rng(1000*k + int(T))
            panel = dict(sp=sp, k=k, typ=case_type(sp, k), T=T, P=P[live], w=w[live], rem=float(w[live].sum()/len(w)), pos0=pos0, flights={})
            for v in VARIANTS:
                sp_pose = start_point(v, P[live], w[live], env.half, rng, pose)
                if sp_pose is None:
                    continue
                obs, info = env.reset(options=case.options(scen))       # rebuild the state at T: replay the first flight's actions
                for a in acts[:n_pre]:
                    obs, r, term, trunc, info = env.step(np.array([a], np.float32))
                if v != "kept":
                    e.plant.reset(np.asarray(sp_pose, float))
                    e.chi_tgt = float(sp_pose[2])
                    obs = env.observation()
                pos_start = float(info["POS"])
                xy, yaw, pos_t = [e.plant.pose[:2].copy()], [float(e.plant.pose[2])], [pos_start]
                term = trunc = False
                while not (term or trunc):
                    a, _ = model.predict(obs, deterministic=True)
                    obs, r, term, trunc, info = env.step(a)
                    xy.append(e.plant.pose[:2].copy()); yaw.append(float(e.plant.pose[2])); pos_t.append(float(info["POS"]))
                circ = circling(yaw)
                minutes_left = (len(yaw) - 1)*5/60
                rows.append(dict(case=f"{'바스토' if sp == 'bval' else '암보이'} {k + 1}번", typ=case_type(sp, k), T=T, variant=v,
                                 start_dist_km=float(np.hypot(P[live][:, 0] - sp_pose[0], P[live][:, 1] - sp_pose[1]).min()/1000),
                                 remaining=float(w[live].sum()/len(w)), gained=pos_t[-1] - pos_start, minutes=minutes_left, circling=float(circ.mean())))
                panel["flights"][v] = (np.array(xy), circ, pos_t[-1] - pos_start)
            panels.append(panel)
            print(f"{sp} k{k} T={T:.0f}: remaining {w[live].sum()/len(w):.3f}, " + ", ".join(
                f"{VN[v]} +{panel['flights'][v][2]:.3f}" for v in panel["flights"]), flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/f"midstart_{args.label}.json").write_text(json.dumps(rows))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["AppleGothic", "NanumGothic", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False
    INK, MUTED, GRID, BG = "#1f1f1e", "#6b6b67", "#e4e3de", "#fcfcfb"
    TN = {"road": "도로형", "mixed": "혼합형", "desert": "사막형"}
    ncol = len(panels)//2 + len(panels) % 2
    fig, axs = plt.subplots(2, ncol, figsize=(5.4*ncol, 11), facecolor=BG, squeeze=False)
    for ax, p in zip(axs.T.ravel(), panels):
        P, w = p["P"], p["w"]
        s = 1 + 40*w/w.max()
        ax.scatter(P[:, 0]/1000, P[:, 1]/1000, s=s, c="#c9772e", alpha=0.35, lw=0, zorder=1)
        allxy = [P/1000]
        for v, (xy, circ, g) in p["flights"].items():
            xy = xy/1000
            allxy.append(xy)
            ax.plot(xy[:, 0], xy[:, 1], color=VC[v], lw=1.3, zorder=3, label=f"{VN[v]}: +{g:.3f}")
            ax.plot(xy[circ, 0], xy[circ, 1], ".", color=VC[v], ms=2.5, zorder=4)
            ax.plot(*xy[0], marker="o", color=VC[v], ms=8, mec="white", zorder=5)
        A = np.vstack(allxy)
        lo, hi = np.percentile(A, 1, axis=0), np.percentile(A, 99, axis=0)
        c, h = (lo + hi)/2, max((hi - lo).max()/2 + 1.5, 4)
        ax.set_xlim(c[0] - h, c[0] + h); ax.set_ylim(c[1] - h, c[1] + h); ax.set_aspect("equal")
        cs = f"{'바스토' if p['sp'] == 'bval' else '암보이'} {p['k'] + 1}번 {TN[p['typ']]}"
        ax.set_title(f"{cs} · 이륙 뒤 {p['T']:.0f}분에 다시 시작\n그때까지 찾은 POS {p['pos0']:.2f}, 남은 확률 {p['rem']:.2f} (주황 = 남은 입자, 클수록 무거움)",
                     loc="left", fontsize=9.5, color=INK)
        ax.legend(fontsize=8, frameon=False, loc="lower left", title="시작점: 남은 시간에 더 찾은 POS", title_fontsize=8)
        ax.tick_params(colors=MUTED, labelsize=7.5); ax.set_facecolor(BG)
        for sp_ in ax.spines.values():
            sp_.set_color(GRID)
    for ax in axs.T.ravel()[len(panels):]:
        ax.axis("off")
    fig.suptitle(f"비행 중 시작 확인 ({args.label}): 첫 비행으로 훑은 믿음에서 기체를 네 곳에 두고 남은 비행을 날림 (동그라미 = 시작점, 굵은 점 = 원 선회 중)",
                 x=0.01, ha="left", fontsize=12.5, color=INK)
    fig.tight_layout()
    fig.savefig(args.out/f"midstart_{args.label}.png", dpi=90, facecolor=BG)
    L = [f"# 비행 중 시작 확인: {args.label}", "", "| 사례 | 유형 | 다시 시작 (이륙 뒤 분) | 시작점 | 가장 가까운 남은 입자까지 (km) | 남은 확률 | 남은 시간 (분) | 더 찾은 POS | 원 선회 몫 |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['case']} | {TN[r['typ']]} | {r['T']:.0f} | {VN[r['variant']]} | {r['start_dist_km']:.1f} | {r['remaining']:.3f} | {r['minutes']:.0f} | "
                 f"{r['gained']:.3f} | {100*r['circling']:.0f}% |")
    (args.out/f"midstart_{args.label}.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()

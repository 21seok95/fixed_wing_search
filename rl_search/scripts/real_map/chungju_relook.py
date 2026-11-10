"""Can a policy deciding every 5 s look away from a ground point and back, so that the pass rule counts a new pass?

Geometry only (open ground, no particles, no belief): every sequence of 8 decisions (-1, 0, +1 x the turn limit, 3^8)
is flown on a copy of the plant from level flight; the forward camera on the 2-axis gimbal is projected on a grid
around the UAV. For a pass gap G a point gets a second pass when it is in view, out of view for more than G seconds,
and in view again (rule of ``detection.PassCap``). Targets: fixed, or moving at 45 km/h along / against / across
the first heading.

Figure: rl_search/data/real/chungju/relook.png, numbers in relook.json

    python -m rl_search.scripts.real_map.chungju_relook
"""
from __future__ import annotations

import copy
import itertools
import json
import warnings

import numpy as np

from rl_search.real_terrain import DATA

GAPS = (2, 5, 10, 15, 30)
N_DEC = 8
CELL = 50.0
V_T = 12.5
TARGETS = {"정지": (0.0, 0.0), "같은 방향 45 km/h": (V_T, 0.0), "반대 방향 45 km/h": (-V_T, 0.0), "옆 방향 45 km/h": (0.0, V_T)}
BATCH = 300


def fly(env, seqs):
    """1 s samples (x, y, yaw, phi) of every action sequence: (K, T, 4)."""
    p0 = copy.deepcopy(env.plant)
    p0.reset(np.array([0.0, 0.0, 0.0]))
    s0 = copy.deepcopy(p0.get_state())
    out = []
    for seq in seqs:
        p0.set_state(copy.deepcopy(s0))
        chi, rows = 0.0, []
        for a in seq:
            chi += a*env.turn
            hist = []
            p0._integrate(chi, p0.n_inner, env.v, out=hist)
            rows += [hist[i][:4] for i in env.sample_idx]
        out.append(rows)
    return np.array(out)


def count_passes(env, poses, vt):
    """Passes per grid point for each gap: dict G -> (K, N) int array, in batches."""
    from rl_search.detection import Gimbal, image_coords
    cfg = env.detector.cfg
    gx = np.arange(-2000.0, 3000.0, CELL) + CELL/2
    gy = np.arange(-2500.0, 2500.0, CELL) + CELL/2
    X, Y = np.meshgrid(gx, gy)
    px0, py0 = X.ravel(), Y.ravel()
    K, T = poses.shape[:2]
    res = {G: np.zeros((K, len(px0)), np.int16) for G in GAPS}
    for b in range(0, K, BATCH):
        P = poses[b:b + BATCH]
        M = len(P)
        gim = Gimbal(M, cfg, yaw0=P[:, 0, 2])
        out = np.full((M, len(px0)), 1e6)
        for t in range(T):
            x, y, yaw, phi = P[:, t].T
            frame, _ = gim.step(yaw, phi, gim.los_forward(yaw), 1.0)
            px, py = px0 + vt[0]*(t + 1), py0 + vt[1]*(t + 1)
            uu, vv, front = image_coords(px, py, x, y, frame, cfg)
            ins = front & (np.abs(uu) <= 1) & (np.abs(vv) <= 1)
            for G in GAPS:
                res[G][b:b + M] += ins & (out > G)
            out = np.where(ins, 0.0, out + 1.0)
    return res


def main():
    warnings.filterwarnings("ignore")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from rl_search.real_scenario import BANK_DEG, make_env
    plt.rcParams["font.family"] = ["AppleGothic", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False
    seqs = np.array(list(itertools.product((-1.0, 0.0, 1.0), repeat=N_DEC)))
    cell = (CELL/1000)**2
    out = {}
    show = None
    for name, bank in (("결정당 30° (RL 기본)", None), ("결정당 49° (롤 30°)", BANK_DEG)):
        env = make_env(0, bank_deg=bank)
        poses = fly(env, seqs)
        out[name] = {}
        for tn, vt in TARGETS.items():
            r = count_passes(env, poses, vt)
            cov = (r[2] >= 1).sum(1)*cell
            row = {}
            for G in GAPS:
                a = (r[G] >= 2).sum(1)*cell
                row[G] = dict(any=float((a > 0).mean()), best=float(a.max()), mean=float(a.mean()),
                              best_seq=seqs[int(a.argmax())].tolist())
            out[name][tn] = dict(cover_mean=float(cov.mean()), by_gap=row)
            print(name, tn, f"cover {cov.mean():.2f} km2", {G: (round(v['any'], 3), round(v['best'], 3)) for G, v in row.items()}, flush=True)
            if bank is not None and tn == "정지":
                show = (env, poses, r)

    fig, axs = plt.subplots(2, 4, figsize=(22, 10), layout="constrained")
    cols = {"정지": "#2a4d7a", "같은 방향 45 km/h": "#c0392b", "반대 방향 45 km/h": "#27ae60", "옆 방향 45 km/h": "#8e44ad"}
    for j, (name, d) in enumerate(out.items()):
        ax = axs[j, 0]
        x = np.arange(len(GAPS)); w = .2
        for k, (tn, v) in enumerate(d.items()):
            ax.bar(x + (k - 1.5)*w, [100*v["by_gap"][G]["any"] for G in GAPS], w, color=cols[tn], label=tn)
        ax.set_xticks(x); ax.set_xticklabels([f"{g}초" for g in GAPS]); ax.set_ylabel("%"); ax.set_ylim(0, 100)
        ax.set_title(f"{name}\n다시 본 땅이 생기는 행동 조합의 비율 (40초, 3⁸개)", fontsize=10); ax.legend(fontsize=8)
        ax = axs[j, 1]
        for k, (tn, v) in enumerate(d.items()):
            ax.bar(x + (k - 1.5)*w, [v["by_gap"][G]["mean"] for G in GAPS], w, color=cols[tn], label=tn)
        ax.set_xticks(x); ax.set_xticklabels([f"{g}초" for g in GAPS]); ax.set_ylabel("km²")
        ax.set_title("다시 본 면적 (조합 평균)", fontsize=10)
        ax = axs[j, 2]
        for k, (tn, v) in enumerate(d.items()):
            ax.bar(x + (k - 1.5)*w, [v["by_gap"][G]["best"] for G in GAPS], w, color=cols[tn], label=tn)
        ax.set_xticks(x); ax.set_xticklabels([f"{g}초" for g in GAPS]); ax.set_ylabel("km²")
        ax.set_title("다시 본 면적 (가장 많이 만드는 조합)", fontsize=10)
    env, poses, r = show
    bank_name = "결정당 49° (롤 30°)"
    for j, G in enumerate((5, 15)):
        ax = axs[j, 3]
        i = int(((r[G] >= 2).sum(1)).argmax())
        ax.plot(poses[i, :, 0]/1000, poses[i, :, 1]/1000, color="#2a4d7a", lw=2)
        ax.plot(0, 0, "o", color="k")
        gx = np.arange(-2000.0, 3000.0, CELL) + CELL/2; gy = np.arange(-2500.0, 2500.0, CELL) + CELL/2
        Z = r[G][i].reshape(len(gy), len(gx))
        ax.imshow(np.where(Z > 0, Z, np.nan), origin="lower", extent=np.array([gx[0], gx[-1], gy[0], gy[-1]])/1000,
                  cmap="YlOrRd", vmin=0, vmax=3, interpolation="nearest")
        ax.set_aspect("equal"); ax.set_xlabel("km"); ax.set_ylabel("km")
        ax.set_title(f"다시 본 면적이 가장 큰 조합 (간격 {G}초, {bank_name}, 정지 표적)\n색 = 통과 횟수, 파란 선 = 무인기, 행동 {seqs[i].astype(int).tolist()}", fontsize=9)
    fig.suptitle("5초마다 결정하는 정책이 같은 땅을 다시 볼 수 있나: 통과 간격별 (전방 45° 카메라, 40초 비행, 땅은 모두 열린 땅)", fontsize=13)
    p = DATA/"chungju"/"relook.png"
    fig.savefig(p, dpi=95)
    p.with_suffix(".json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"Saved {p}")


if __name__ == "__main__":
    main()

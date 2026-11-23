"""The two teammate channels agreed with the user on 2026-11-20 (to replace the M1 tail channel and teammate vector), drawn before they are built:

- ④ 동료 영역 (every map), two forms: hard = 1 where a teammate is nearer than this UAV, else 0; soft = d_me / (d_me + d_mate) (0 at me,
  0.5 where me and the nearest teammate are as near, 1 at the teammate), straight distances from the positions now
- 위험 구역 (the 1.6 / 3.2 / 6.4 km maps): where each teammate can be within that map's lead time (15 / 30 / 60 s), widened by the penalty
  distance 250 m, brighter = sooner (1 now .. 0.3 at the lead). "Can be": the teammate's own flight model flown with the policy's commands
  (eight command sizes, left and right, for 0..all decisions of the lead, then straight on), so the fan follows its heading, its turn state
  and its turn limit (a full command turns the course target 49 deg per 5 s decision, about 10 deg/s). One 60 s fan per teammate and step
  (step 0.25 s: at most 13 m from the 0.1 s flight model), shared by every UAV that looks at it; the 15 / 30 s fans are its first seconds.

at two moments of a team flight (Jirisan validation case, a model, deterministic), for two UAVs, with the time per UAV and step.

    python -m rl_search.scripts.multi_design.mate_channels_view --model rl_search/data/rl_real/m1_multi/ckpt/model_1200015.zip --label m1_1.2M
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np

from rl_search.scripts.multi_design.ma_eval import OUT, RUN_CFG, _gaps

D_SOFT = 250.0                     # penalty distance (user 2026-11-20)
DANGER_MAPS = (0, 1, 2)            # 1.6, 3.2, 6.4 km
FRACS = (-1.0, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 1.0)    # command sizes of the turns in the fan (straight on is the n = 0 path)
REACH_H = 0.25                     # integration step of the fan (s)
MOMENTS = ((3.0, "줄지어 이동 중"), (35.0, "뭉쳐 맴도는 중"))


def reach_points(plant, chi_tgt: float, turn: float, dec_s: float, T: float, h: float = REACH_H, every_s: float = 1.0):
    """Where a UAV can be within T s: its flight model (``plants.FixedWing`` equations, as ``rl_multi.predict_paths``) flown with a command
    of each size in FRACS for the first n decisions (n = 0 .. T / dec_s) and straight on after. Returns points (M, 2) and their times (M,)."""
    from rl_search.plants import G
    n_dec = int(np.ceil(T/dec_s))
    seqs = [np.zeros(n_dec)] + [np.r_[np.full(n, f), np.zeros(n_dec - n)] for f in FRACS for n in range(1, n_dec + 1)]
    A = np.array(seqs)                                             # (P, n_dec) commands
    chi_seq = chi_tgt + np.cumsum(A, axis=1)*turn                  # each command moves the course target by a x turn
    m = len(A)
    v = plant.v
    n, k, per_dec = int(round(T/h)), max(1, int(round(every_s/h))), int(round(dec_s/h))
    x, y, yaw = (np.full(m, float(c)) for c in plant.pose)
    phi, r = np.full(m, float(plant.phi)), np.full(m, float(plant.r))
    w2, c2 = plant.omega**2, 2*plant.zeta*plant.omega
    pts, ts = [np.array([plant.pose[:2]])], [np.zeros(1)]
    for i in range(n):
        chi = chi_seq[:, min(i//per_dec, n_dec - 1)]
        err = (chi - yaw + np.pi) % (2*np.pi) - np.pi
        r = r + (w2*err - c2*r)*h
        phi_cmd = np.clip(np.arctan(r*v/G), -plant.phi_max, plant.phi_max)
        phi = phi + np.clip((phi_cmd - phi)/plant.tau, -plant.p_max, plant.p_max)*h
        yaw = yaw + G*np.tan(phi)/v*h
        x, y = x + v*np.cos(yaw)*h, y + v*np.sin(yaw)*h
        if (i + 1) % k == 0:
            pts.append(np.stack([x, y], 1))
            ts.append(np.full(m, (i + 1)*h))
    return np.concatenate(pts), np.concatenate(ts)


def _cells(x, y, yaw, side, px):
    cell = side/px
    c = (np.arange(px) + 0.5)*cell
    F, R = np.meshgrid(side/2 - c, c - side/2, indexing="ij")
    cs, sn = np.cos(yaw), np.sin(yaw)
    return F, R, x + F*cs + R*sn, y + F*sn - R*cs


def territory_map(x, y, yaw, mates_xy, side, px=64, soft=False) -> np.ndarray:
    """④: hard = 1 where the nearest teammate is nearer than this UAV; soft = d_me / (d_me + d_mate). Cell centres, straight distance."""
    if not len(mates_xy):
        return np.zeros((px, px), np.float32)
    F, R, WX, WY = _cells(x, y, yaw, side, px)
    d2_m = np.min([(WX - mx)**2 + (WY - my)**2 for mx, my in mates_xy], axis=0)
    d2_me = F*F + R*R
    if soft:
        d_me, d_m = np.sqrt(d2_me), np.sqrt(d2_m)
        return (d_me/np.maximum(d_me + d_m, 1e-9)).astype(np.float32)
    return (d2_m < d2_me).astype(np.float32)


def disk_max(g: np.ndarray, rc: int) -> np.ndarray:
    """Largest value within a disk of radius rc cells around every cell: a 1-D max filter along the rows of the half-width of each disk row,
    shifted up and down (the same result as ``grey_dilation`` with a disk footprint, about 12 times faster for rc 10)."""
    from scipy.ndimage import maximum_filter1d
    out = np.zeros_like(g)
    H = g.shape[0]
    rows = {}
    for dy in range(-rc, rc + 1):
        w = int(np.floor(np.sqrt(rc*rc - dy*dy)))
        if w not in rows:
            rows[w] = maximum_filter1d(g, size=2*w + 1, axis=1, mode="constant")
        m = rows[w]
        if dy >= 0:
            out[dy:] = np.maximum(out[dy:], m[:H - dy])
        else:
            out[:dy] = np.maximum(out[:dy], m[-dy:])
    return out


def danger_map(x, y, yaw, pts, vals, side, px=64, d=D_SOFT) -> np.ndarray:
    """Teammates' reachable points (world) with values, on a heading-up map, each spread over a disk of radius d (largest value per cell)."""
    cell = side/px
    rc = int(np.ceil(d/cell))
    g = np.zeros((px + 2*rc, px + 2*rc), np.float32)
    if len(pts):
        dd = pts - np.array([x, y])
        cs, sn = np.cos(yaw), np.sin(yaw)
        fwd, right = dd[:, 0]*cs + dd[:, 1]*sn, dd[:, 0]*sn - dd[:, 1]*cs
        row = np.floor((side/2 - fwd)/cell).astype(int) + rc
        col = np.floor((right + side/2)/cell).astype(int) + rc
        ok = (row >= 0) & (row < g.shape[0]) & (col >= 0) & (col < g.shape[1])
        np.maximum.at(g, (row[ok], col[ok]), vals[ok].astype(np.float32))
        g = disk_max(g, int(np.floor(d/cell)))
    return g[rc:rc + px, rc:rc + px]


def reach_all(env) -> dict:
    """One fan per active UAV for the longest danger lead (shared by every UAV that looks at it)."""
    T = max(float(env.maps[k][2]) for k in DANGER_MAPS)
    out = {}
    for j in np.flatnonzero(env.active):
        ej = env.uavs[j].env
        out[int(j)] = reach_points(ej.plant, ej.chi_tgt, ej.turn, float(ej.cfg.decision_dt_s), T)
    return out


def channels(env, i: int, reach: dict, both: bool = True):
    """UAV i's ④ (hard and soft; ``both`` False: soft only, as it would be built) on every map and danger on DANGER_MAPS, from the env
    state now and the shared fans."""
    x, y, yaw = env.uavs[i].env.plant.pose
    mates = [j for j in reach if j != i]
    mates_xy = [tuple(env.uavs[j].env.plant.pose[:2]) for j in mates]
    hard = [territory_map(x, y, yaw, mates_xy, side) for _, side, _ in env.maps] if both else None
    soft = [territory_map(x, y, yaw, mates_xy, side, soft=True) for _, side, _ in env.maps]
    dang = {}
    for k in DANGER_MAPS:
        _, side, lead = env.maps[k]
        P, V = [], []
        for j in mates:
            p, t = reach[j]
            m = t <= float(lead) + 1e-9
            P.append(p[m])
            V.append(1.0 - 0.7*t[m]/float(lead))
        dang[k] = danger_map(x, y, yaw, np.concatenate(P) if P else np.zeros((0, 2)), np.concatenate(V) if V else np.zeros(0), side)
    return hard, soft, dang


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--case", type=int, default=0)
    ap.add_argument("--pair", type=int, nargs=2, default=[1, 0], help="UAVs drawn (0 = first launched); the first gets all seven maps")
    a = ap.parse_args()
    warnings.filterwarnings("ignore")
    import torch
    from stable_baselines3 import PPO
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from rl_search.rl_multi import MultiRealEnv
    from rl_search.rl_real import RealRLConfig
    from rl_search.rl_real_eval import _case
    from rl_search.scripts.rl_eval.signal_check import settings_of
    plt.rcParams["font.family"] = "AppleGothic"
    plt.rcParams["axes.unicode_minus"] = False
    torch.set_num_threads(1)
    c = json.loads(Path(RUN_CFG).read_text())
    s = settings_of(c) | ({"look_w": float(c["look_w"])} if c.get("look_w") else {})
    model = PPO.load(a.model, device="cpu")
    case, scen, _, _ = _case("v2j", 3, a.case)
    env = MultiRealEnv(a.k, RealRLConfig(region="jirisan120", settings=s, mask_cost=float(c.get("mask_cost") or 0.0)), seed=a.case)
    obs = env.reset(options=case.options(scen), gaps=_gaps(a.case, a.k))
    t0 = env.T
    cols = ["#2a6fdb", "#e07b39", "#3a9a5b", "#8e44ad"]
    i_main, i_other = a.pair
    fig, axs = plt.subplots(3*len(MOMENTS), 8, figsize=(30, 11.5*len(MOMENTS)))
    timing = []
    for r, (minute, what) in enumerate(MOMENTS):
        while (env.T - t0)/60.0 < minute - 1e-9 and not env.done:
            batch = {key: np.stack([o[key] for o in obs]) for key in obs[0]}
            acts, _ = model.predict(batch, deterministic=True)
            obs, _, _, _ = env.step([np.atleast_1d(x) for x in np.ravel(acts)])
        n_act = int(env.active.sum())
        t1 = time.perf_counter()
        for _ in range(10):                                        # what one 5 s step costs: the fans once, then every UAV's channels
            rr = reach_all(env)
            for i in np.flatnonzero(env.active):
                channels(env, int(i), rr, both=False)
        timing.append((time.perf_counter() - t1)/10*1000/n_act)
        reach = reach_all(env)
        hard, soft, dang = channels(env, i_main, reach)
        _, _, dang_o = channels(env, i_other, reach)
        x, y, yaw = env.uavs[i_main].env.plant.pose
        mates = [j for j in reach if j != i_main]

        def marks(ax, side, me=i_main, north=False):
            px_, py_, pyaw = env.uavs[me].env.plant.pose
            fy = np.pi/2 if north else pyaw
            for j in range(a.k):
                if j == me or not env.active[j]:
                    continue
                d = env.uavs[j].env.plant.pose[:2] - np.array([px_, py_])
                fwd, right = d[0]*np.cos(fy) + d[1]*np.sin(fy), d[0]*np.sin(fy) - d[1]*np.cos(fy)
                if abs(fwd) <= side/2 and abs(right) <= side/2:
                    ax.plot(right/1000, fwd/1000, "o", mfc="none", mec=cols[j], ms=10, mew=1.8)
            ax.plot(0, 0, marker=(3, 0, (0 if not north else np.degrees(pyaw - np.pi/2))), color="#00b4d8", ms=11)

        R0 = 3*r
        # row 1: top-down (north up) of the 3.2 km area with the teammates' 30 s fans, then hard ④ on all seven maps
        ax = axs[R0, 0]
        _, side, lead = env.maps[1]
        P = [reach[j][0][reach[j][1] <= lead] for j in mates]
        V = [1.0 - 0.7*reach[j][1][reach[j][1] <= lead]/lead for j in mates]
        north = danger_map(x, y, np.pi/2, np.concatenate(P), np.concatenate(V), side) if mates else np.zeros((64, 64))
        ax.imshow(north, cmap="Reds", vmin=0, vmax=1, extent=(-side/2000, side/2000, -side/2000, side/2000), interpolation="nearest")
        for j, p in zip(mates, P):
            d = p - np.array([x, y])
            ax.scatter(d[:, 0]/1000, d[:, 1]/1000, s=1, c=cols[j], alpha=0.4)
        marks(ax, side, north=True)
        ax.set_xlim(-side/2000, side/2000)
        ax.set_ylim(-side/2000, side/2000)
        ax.set_aspect("equal")
        ax.set_title(f"위에서 본 그림 (북쪽 위), {side/1000:g} km 둘레\n점 = 동료가 {lead:.0f}초 안에 갈 수 있는 곳, 빨강 = 그 250 m 안", fontsize=9)
        ax.set_ylabel(f"{minute:g}분: {what}\n{i_main + 1}번째 기체 기준", fontsize=12)
        for row, maps_, cmap, lab in ((R0, hard, "Blues", "④ 딱 자름"), (R0 + 1, soft, "Blues", "④ 부드럽게")):
            for k, (name, side, lead) in enumerate(env.maps):
                ax = axs[row, 1 + k]
                ax.imshow(maps_[k], cmap=cmap, vmin=0, vmax=1, extent=(-side/2000, side/2000, -side/2000, side/2000), interpolation="nearest")
                marks(ax, side)
                ax.set_xticks([])
                ax.set_yticks([])
                extra = f"흰 = 내 몫 {100*(1 - maps_[k].mean()):.0f}%" if row == R0 else f"0.4~0.6(비슷) {100*((maps_[k] > 0.4) & (maps_[k] < 0.6)).mean():.0f}%"
                ax.set_title(f"{lab} · {side/1000:g} km\n({extra})", fontsize=9)
        ax = axs[R0 + 1, 0]
        ax.axis("off")
        ax.text(0.0, 0.5, "④ 딱 자름: 칸마다 나보다 동료가\n   가까우면 1(파랑), 아니면 0(흰)\n\n④ 부드럽게: 칸마다\n   (나까지 거리) ÷ (나까지 + 가장 가까운\n   동료까지 거리)\n   0 = 나만 가까움, 0.5 = 비슷,\n   1 = 동료만 가까움\n   → 다 같이 붙어 있으면 큰 지도는\n      거의 0.5(아직 안 나뉨)",
                fontsize=10, va="center", transform=ax.transAxes)
        # row 3: danger on the three small maps for the two UAVs
        ax = axs[R0 + 2, 0]
        ax.axis("off")
        ax.text(0.0, 0.5, f"위험 구역 채널\n= 동료가 지도의 예측 시간\n   (15 / 30 / 60초) 안에 갈 수\n   있는 곳 + 250 m\n밝을수록 빨리 닿음\n\n왼쪽 3장: {i_main + 1}번째 기체\n오른쪽 3장: {i_other + 1}번째 기체\n\n계산 시간 (기체 한 대, 한 걸음:\n부드러운 ④ 7장 + 위험 3장\n+ 비행 반경):\n{timing[-1]:.1f} ms (떠 있는 기체 {n_act}대)",
                fontsize=10, va="center", transform=ax.transAxes)
        for cc, (who, dd) in enumerate(((i_main, dang), (i_other, dang_o))):
            for q, k in enumerate(DANGER_MAPS):
                _, side, lead = env.maps[k]
                ax = axs[R0 + 2, 1 + cc*4 + q]
                ax.imshow(dd[k], cmap="Reds", vmin=0, vmax=1, extent=(-side/2000, side/2000, -side/2000, side/2000), interpolation="nearest")
                marks(ax, side, me=who)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(f"위험 구역 · {who + 1}번째 기체 · {side/1000:g} km ({lead:.0f}초)\n켜진 칸 {100*(dd[k] > 0).mean():.0f}%", fontsize=9)
        axs[R0 + 2, 4].axis("off")
    fig.suptitle(f"새 동료 채널 미리 보기: {a.label}, 지리산 사례 {a.case + 1}, {a.k}대. 모든 지도 위 = 기수 방향(맨 왼쪽 위 그림만 북쪽 위), 하늘색 삼각형 = 자기, "
                 f"동그라미 = 동료 실제 위치 (1번째 파랑, 2번째 주황, 3번째 초록, 4번째 보라)", fontsize=13)
    path = OUT/f"mate_channels_{a.label}_k{a.k}.png"
    fig.savefig(path, dpi=75, bbox_inches="tight")
    print(path, "ms per UAV and step:", [round(t, 2) for t in timing])


if __name__ == "__main__":
    main()

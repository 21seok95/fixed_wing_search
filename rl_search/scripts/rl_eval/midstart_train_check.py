"""Checks of the mid-flight starts in training (start stage 4, rl_real.MID_KINDS, user 2026-11-17) on the redesigned scenarios (V2 pools of
the 120 km maps), flown by R12 (6M, deterministic) as a stand-in policy:

  exact  - a launch flight saves its state; a "kept" start from that state must fly on exactly as the flight did (every step's position,
           reward, POS and observation); then the five start kinds are drawn 60 times each on that saved state (no flying)
  loop   - episodes as training draws them (no options): launch / mid-flight shares, the saved times, the start kinds, distances, what
           R12 finds after each start, memory
  figure - report/midstart_train_check.png and .md

    python -m rl_search.scripts.rl_eval.midstart_train_check exact amboy120
    python -m rl_search.scripts.rl_eval.midstart_train_check loop amboy120 --episodes 40
    python -m rl_search.scripts.rl_eval.midstart_train_check figure
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
import warnings
from pathlib import Path

import numpy as np

REGIONS = ("amboy120", "mojave120")
NAME = {"amboy120": "암보이 120 km", "mojave120": "바스토 120 km"}
OUT = Path("rl_search/data/real/amboy/rl_eval/report")
WORK = OUT/"midstart_train"
MODEL = "rl_search/data/rl_real/r12_2map/ckpt/model_6000000.zip"
KN = {"kept": "그대로", "heavy": "무거운 덩어리 안", "empty": "거의 빈 곳", "edge": "가장자리 바깥 3~6 km", "far": "먼 곳 10~15 km"}
KC = {"kept": "#1f1f1e", "heavy": "#2a78d6", "empty": "#d6336c", "edge": "#eb6834", "far": "#1baf7a"}
TN = {"road": "도로형", "mixed": "혼합형", "desert": "사막형"}


def rss_mb() -> float:
    return int(subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())], capture_output=True, text=True).stdout.strip())/1024


def make(region: str, seed: int):
    from rl_search.rl_real import RealRLConfig, RealRLEnv, pool_dir_multi
    from rl_search.scripts.rl_eval.signal_check import settings_of
    settings = settings_of(json.loads(Path("rl_search/data/rl_real/r12_2map/config.json").read_text())) | dict(scen_v2=True)
    return RealRLEnv(RealRLConfig(region=region, multi=True, pool=pool_dir_multi(settings, region), settings=settings, stage=3), seed=seed)


def ohash(obs) -> str:
    h = hashlib.sha1()
    for k in sorted(obs):
        h.update(np.ascontiguousarray(obs[k]).tobytes())
    return h.hexdigest()


def fly(env, model, obs, rec: bool = True):
    """Fly to the end of the flight: per step (t, x, y, yaw, reward, dPOS, POS) and the observation hashes."""
    rows, hashes = [], [ohash(obs)]
    term = trunc = False
    while not (term or trunc):
        a, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, info = env.step(a)
        e = env.env
        rows.append((e.t, *e.plant.pose, float(r), float(info["dPOS"]), float(info["POS"]), len(env._mid)))
        hashes.append(ohash(obs))
    return np.array(rows), hashes


def exact(region: str, seed: int):
    warnings.filterwarnings("ignore")
    from rl_search.rl_real import MID_KINDS
    from rl_search.rl_real_eval import _model
    model = _model(MODEL)
    env = make(region, seed)
    obs, info = env.reset()
    assert env.start_kind == "launch", env.start_kind
    e = env.env
    launch = [(float(t), float(x), float(y)) for t, x, y in [env.transit_start[:3]]] + [(t, x, y) for t, x, y, _ in env.transit_track]
    t0 = time.perf_counter()
    rows, hashes = fly(env, model, obs)
    sec_per_step = (time.perf_counter() - t0)/len(rows)
    k = int(np.flatnonzero(rows[:, 7] > 0)[0])               # the step after which the state was saved
    m = env._mid[0]
    t_s = time.perf_counter()
    obs2, info2 = env._mid_start(m=m, kind="kept")
    restore_ms = 1000*(time.perf_counter() - t_s)
    remaining = float(env.start_info["remaining"])          # at the saved time
    rows2, hashes2 = fly(env, model, obs2)
    a, b = rows[k + 1:], rows2
    n = min(len(a), len(b))
    res = dict(region=region, seed=seed, typ=m["typ"], save_min=m["save_min"], save_step=k, steps_after=len(a), steps_kept=len(b),
               same_length=len(a) == len(b), max_pos_m=float(np.abs(a[:n, 1:3] - b[:n, 1:3]).max()), max_reward=float(np.abs(a[:n, 4] - b[:n, 4]).max()),
               max_dpos=float(np.abs(a[:n, 5] - b[:n, 5]).max()), same_obs=hashes[k + 1:] == hashes2,
               pos_after_flight=float(a[:, 5].sum()), pos_after_kept=float(b[:, 5].sum()), sec_per_step=sec_per_step, restore_ms=restore_ms,
               remaining=remaining)
    print(json.dumps(res, indent=1), flush=True)
    # the five kinds, drawn on this saved state (the state stays as restored: _mid_point only reads it)
    env._mid_start(m=m, kind="kept")
    b_ = env.env.belief
    P, p = b_.positions(env.env.t), b_.w/b_.n
    pts = {}
    t_s = time.perf_counter()
    for kind in MID_KINDS[1:]:
        pts[kind] = [list(map(float, x)) if x is not None else None for x in (env._mid_point(kind) for _ in range(60))]
    res["draw_ms"] = 1000*(time.perf_counter() - t_s)/(60*4)
    res["kept_pose"] = list(map(float, m["snap"]["plant"]["pose"]))
    WORK.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(WORK/f"exact_{region}.npz", P=P.astype(np.float32), p=p, flight=rows, kept=rows2, launch=np.array(launch),
                        lkp=np.asarray(m["scen"].lkp, float))
    (WORK/f"exact_{region}.json").write_text(json.dumps(res | dict(points=pts)))


def _arrays(x):
    if isinstance(x, np.ndarray):
        yield x
    elif isinstance(x, dict):
        for v in x.values():
            yield from _arrays(v)
    elif isinstance(x, (list, tuple)):
        for v in x:
            yield from _arrays(v)


def mid_memory(env) -> tuple:
    """Memory the saved states add to one env: their own arrays (MB), and the scenarios only they keep alive (count, MB of tracks)."""
    own = sum(a.nbytes for m in env._mid for a in _arrays(m["snap"]))/1e6
    cur = id(env._scen.track) if env._scen is not None else None
    extra = {id(m["scen"].track): m["scen"].track.nbytes for m in env._mid if id(m["scen"].track) != cur}
    return own, len(extra), sum(extra.values())/1e6


def loop(region: str, episodes: int, seed: int):
    warnings.filterwarnings("ignore")
    from rl_search.rl_real_eval import _model
    model = _model(MODEL)
    env = make(region, seed)
    out, t0 = [], time.perf_counter()
    for ep in range(episodes):
        buf = len(env._mid)
        last = id(env._mid[-1]) if env._mid else None
        obs, info = env.reset()
        e = env.env
        row = dict(ep=ep, start=env.start_kind, typ=env._ep_typ, buffer=buf, t_start_min=e.t/60, x=float(e.plant.pose[0]), y=float(e.plant.pose[1]),
                   info={k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) and not isinstance(v, bool) else v)
                         for k, v in env.start_info.items()})
        row["remaining_start"] = float(e.belief.mass)
        term = trunc = False
        n = 0
        while not (term or trunc):
            a, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(a)
            n += 1
        own, n_extra, extra_mb = mid_memory(env)
        saved = row["start"] == "launch" and bool(env._mid) and id(env._mid[-1]) != last
        row.update(pos=float(info["POS"]), steps=n, t_end_min=e.t/60, saved=saved,
                   save_min=float(env._mid[-1]["save_min"]) if saved else None, rss_mb=rss_mb(),
                   mid_own_mb=own, mid_extra_scen=n_extra, mid_extra_mb=extra_mb)
        out.append(row)
        print(f"{region} ep {ep}: {row['start']:>10} {row['typ']:>6} buffer {buf} start {row['t_start_min']:.1f} min, POS {row['pos']:.3f}, "
              f"{n} steps, saved at {row['save_min']}, states {own:.2f} MB + {n_extra} scenarios {extra_mb:.0f} MB, {time.perf_counter() - t0:.0f} s",
              flush=True)
        WORK.mkdir(parents=True, exist_ok=True)
        (WORK/f"loop_{region}.json").write_text(json.dumps(out))


def figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from scipy import ndimage
    from rl_search.rl_real import MID_CELL_M, MID_EMPTY, MID_KINDS
    plt.rcParams["font.family"] = ["AppleGothic", "NanumGothic", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False
    INK, MUTED, GRID, BG = "#1f1f1e", "#6b6b67", "#e4e3de", "#fcfcfb"
    ex = {r: (json.loads((WORK/f"exact_{r}.json").read_text()), np.load(WORK/f"exact_{r}.npz")) for r in REGIONS if (WORK/f"exact_{r}.json").exists()}
    lp = {r: json.loads((WORK/f"loop_{r}.json").read_text()) for r in REGIONS if (WORK/f"loop_{r}.json").exists()}
    fig = plt.figure(figsize=(25, 15.5), facecolor=BG)
    gs = fig.add_gridspec(2, 4, height_ratios=[1.15, 1], width_ratios=[1, 1, 0.9, 0.9])
    half = 60000.0 - 12.5
    for col, (r, (res, d)) in enumerate(ex.items()):
        ax = fig.add_subplot(gs[0, col])
        P, p = d["P"].astype(float), d["p"]
        nc = int(np.ceil(2*half/MID_CELL_M))
        ins = (np.abs(P) <= half).all(1) & (p > 0)
        q = np.clip(np.floor((P[ins] + half)/MID_CELL_M).astype(int), 0, nc - 1)
        G = np.zeros((nc, nc))
        np.add.at(G, (q[:, 1], q[:, 0]), p[ins])
        lab, nlab = ndimage.label(G > MID_EMPTY, structure=np.ones((3, 3)))
        sums = ndimage.sum(G, lab, index=np.arange(1, nlab + 1))
        big = int(np.argmax(sums)) + 1
        ext = [-half/1000, half/1000, -half/1000, half/1000]
        im = ax.imshow(np.where(G > 0, G, np.nan), origin="lower", extent=ext, cmap="Oranges", norm=LogNorm(vmin=1e-5, vmax=G.max()))
        ax.contour((np.arange(nc) + .5)*MID_CELL_M/1000 - half/1000, (np.arange(nc) + .5)*MID_CELL_M/1000 - half/1000, (lab == big).astype(float),
                   levels=[.5], colors=KC["heavy"], linewidths=1.8)
        fl = d["flight"]
        k = res["save_step"]
        path = np.vstack([d["launch"][:, 1:3], fl[:k + 1, 1:3]])/1000
        ax.plot(path[:, 0], path[:, 1], color=MUTED, lw=1.1, label=f"이륙부터 저장 시각({res['save_min']:.0f}분)까지 비행")
        ax.plot(*path[0], marker="^", color=INK, ms=9, mec="white", ls="none", label="이륙점")
        for kind in MID_KINDS[1:]:
            xy = np.array([x[:2] for x in res["points"][kind] if x is not None])/1000
            ax.scatter(xy[:, 0], xy[:, 1], s=16, color=KC[kind], ec="white", lw=0.4, zorder=5, label=f"{KN[kind]} (60번 뽑음)")
        ax.plot(*(np.array(res["kept_pose"][:2])/1000), marker="o", ms=11, color=KC["kept"], mec="white", mew=1.5, ls="none", zorder=6,
                label="그대로 (저장 때 위치)")
        allp = np.vstack([path] + [np.array([x[:2] for x in res["points"][kd] if x is not None])/1000 for kd in MID_KINDS[1:]])
        lo, hi = np.percentile(allp, 0.5, axis=0), np.percentile(allp, 99.5, axis=0)
        c, h = (lo + hi)/2, max((hi - lo).max()/2 + 3, 10)
        ax.set_xlim(max(c[0] - h, -60), min(c[0] + h, 60)); ax.set_ylim(max(c[1] - h, -60), min(c[1] + h, 60)); ax.set_aspect("equal")
        ax.set_title(f"{NAME[r]} {TN.get(res['typ'], res['typ'])}: 이륙 뒤 {res['save_min']:.0f}분에 저장한 상태 (남은 확률 {res['remaining']:.2f})\n"
                     f"주황 = 1 km 칸의 남은 확률, 파란 선 = 가장 무거운 덩어리 ({sums[big - 1]/G.sum():.0%}), 점 = 시작점 종류별", loc="left", fontsize=10, color=INK)
        ax.legend(fontsize=7.8, frameon=False, loc="upper left", bbox_to_anchor=(0.0, -0.07), ncol=3, columnspacing=1.0, handletextpad=0.3)
        ax.set_xlabel("동쪽 (km)", fontsize=8.5, color=MUTED, labelpad=1); ax.set_ylabel("북쪽 (km)", fontsize=8.5, color=MUTED)
        if col == 1:
            cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.01)
            cb.set_label("1 km 칸의 남은 확률", fontsize=8.5)
            tk = [x for x in (1e-5, 1e-4, 1e-3, 1e-2, 1e-1) if x <= G.max()]
            cb.set_ticks(tk)
            cb.set_ticklabels([f"{x:g}" for x in tk])
    # exactness: the flight after the save and the kept start
    ax = fig.add_subplot(gs[0, 2])
    for r, (res, d) in ex.items():
        fl, kp = d["flight"], d["kept"]
        k = res["save_step"]
        a = fl[k + 1:]
        ax.plot((a[:, 0] - a[0, 0])/60, np.cumsum(a[:, 5]), color="#2a78d6" if r == REGIONS[0] else "#eb6834", lw=5, alpha=0.35,
                label=f"{NAME[r]}: 원래 비행 (저장 뒤)")
        ax.plot((kp[:, 0] - kp[0, 0])/60, np.cumsum(kp[:, 5]), color="#2a78d6" if r == REGIONS[0] else "#eb6834", lw=1.4, ls="--",
                label=f"{NAME[r]}: 저장 상태에서 '그대로' 시작")
    ax.set_title("확인 1: '그대로' 시작 = 원래 비행이 그대로 이어짐\n(R12, 같은 행동 규칙) 저장 뒤 더 찾은 POS", loc="left", fontsize=10.5, color=INK)
    ax.set_xlabel("저장 뒤 분", fontsize=8.5, color=MUTED)
    ax.legend(fontsize=7.8, frameon=False, loc="upper left")
    txt = []
    for r, (res, d) in ex.items():
        ok = res["same_length"] and res["max_pos_m"] == 0 and res["max_reward"] == 0 and res["same_obs"]
        txt.append(f"{NAME[r]}: 저장 뒤 걸음 {res['steps_after']} / {res['steps_kept']} → {'같음' if ok else '다름'}\n"
                   f"  위치 차 최대 {res['max_pos_m']:.0f} m, 보상 차 {res['max_reward']:.0g}, 관측 {'모두 같음' if res['same_obs'] else '다름'}\n"
                   f"  복원 {res['restore_ms']:.0f} ms, 시작점 뽑기 {res['draw_ms']:.0f} ms, 한 걸음 {1000*res['sec_per_step']:.0f} ms")
    ax.text(0.99, 0.23, "\n".join(txt), transform=ax.transAxes, fontsize=8, color=INK, va="bottom", ha="right",
            bbox=dict(facecolor=BG, edgecolor=GRID, boxstyle="round,pad=0.4"))
    # distances of the drawn start points to the nearest particle, by kind
    ax = fig.add_subplot(gs[0, 3])
    for i, r in enumerate(ex):
        res, d = ex[r]
        P, p = d["P"].astype(float), d["p"]
        from scipy.spatial import cKDTree
        tree = cKDTree(P[p > 0])
        for j, kind in enumerate(MID_KINDS[1:]):
            xy = np.array([x[:2] for x in res["points"][kind] if x is not None])
            dist = tree.query(xy)[0]/1000
            yy = j + (i - 0.5)*0.3 + np.random.default_rng(j).uniform(-0.08, 0.08, len(dist))
            ax.scatter(dist, yy, s=10, color=KC[kind], alpha=0.75, lw=0, marker="o" if i == 0 else "s",
                       label=NAME[r] if j == 0 else None)
    for lo_, hi_, j in ((0, 1, 0), (0, 1, 1), (3, 6, 2), (10, 15, 3)):
        ax.axvspan(lo_, hi_, ymin=(j + 0.1)/4, ymax=(j + 0.9)/4, color=GRID, alpha=0.6, lw=0)
    ax.set_yticks(range(4), [KN[k] for k in MID_KINDS[1:]], fontsize=9)
    ax.set_ylim(-0.5, 3.5); ax.set_xlim(0, 16)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax.set_xlabel("시작점에서 가장 가까운 입자까지 (km)", fontsize=8.5, color=MUTED)
    ax.set_title("확인 2: 종류마다 60번 뽑은 시작점의 거리\n(회색 = 정한 범위; ○ 암보이, □ 바스토)", loc="left", fontsize=10.5, color=INK)
    # training loop
    if lp:
        rows = [x for r in lp for x in lp[r]]
        ax = fig.add_subplot(gs[1, 0])
        names = ["launch"] + [f"mid_{k}" for k in MID_KINDS]
        lab_ = ["이륙"] + [KN[k] for k in MID_KINDS]
        cnt = np.array([sum(x["start"] == s for x in rows) for s in names], float)
        exp = np.array([0.7] + [0.3/5]*5)*len(rows)
        ax.bar(range(len(names)), cnt, color=["#9a9a95"] + [KC[k] for k in MID_KINDS], width=0.62)
        ax.scatter(range(len(names)), exp, marker="_", s=600, color=INK, zorder=4, label="정한 비율의 기댓값 (이륙 70%, 다섯 종류 6%씩)")
        for i, c_ in enumerate(cnt):
            ax.text(i, c_ + 0.4, f"{c_:.0f}", ha="center", fontsize=9, color=INK)
        ax.set_xticks(range(len(names)), lab_, fontsize=8.3, rotation=20)
        n_mid = int(cnt[1:].sum())
        ax.set_title(f"확인 3: 학습처럼 뽑은 에피소드 {len(rows)}개의 시작 (두 지도)\n이륙 {int(cnt[0])} ({cnt[0]/len(rows):.0%}), 비행 중 {n_mid} "
                     f"({n_mid/len(rows):.0%}); 첫 에피소드는 저장 상태가 없어 늘 이륙", loc="left", fontsize=10.5, color=INK)
        ax.legend(fontsize=8, frameon=False)
        ax = fig.add_subplot(gs[1, 1])
        sv = [x["save_min"] for x in rows if x["start"] == "launch" and x["save_min"] is not None]   # only launch flights save (a mid start
        # that used up the last saved state changes the last entry too: not a save)
        used = [x["info"]["save_min"] for x in rows if x["start"].startswith("mid_")]
        ax.hist([sv, used], bins=np.arange(10, 52, 4), color=["#2a78d6", "#eb6834"],
                label=[f"저장한 시각 (이륙 비행 {len(sv)}번)", f"쓴 상태의 저장 시각 (비행 중 시작 {len(used)}번)"])
        ax.set_xlabel("이륙 뒤 분", fontsize=8.5, color=MUTED)
        ax.set_title("저장 시각: 이륙 비행마다 한 번, 10~50분에서 고르게 뽑음", loc="left", fontsize=10.5, color=INK)
        ax.legend(fontsize=8, frameon=False)
        ax = fig.add_subplot(gs[1, 2])
        mids = [x for x in rows if x["start"].startswith("mid_")]
        for kind in MID_KINDS:
            xs = [x for x in mids if x["start"] == f"mid_{kind}"]
            ax.scatter([x["info"]["remaining"] for x in xs], [x["pos"] for x in xs], s=36, color=KC[kind], ec="white", lw=0.5, label=KN[kind])
        ax.set_xlabel("시작 때 남은 확률", fontsize=8.5, color=MUTED); ax.set_ylabel("시작 뒤 R12가 찾은 POS", fontsize=8.5, color=MUTED)
        ax.set_title("비행 중 시작: 남은 확률과 그 뒤 찾은 것 (R12)", loc="left", fontsize=10.5, color=INK)
        ax.legend(fontsize=8, frameon=False)
        ax = fig.add_subplot(gs[1, 3])
        for r, col_ in zip(REGIONS, ("#2a78d6", "#eb6834")):
            if r in lp:
                ep_ = [x["ep"] for x in lp[r]]
                ax.plot(ep_, [x["mid_own_mb"] + x["mid_extra_mb"] for x in lp[r]], color=col_, lw=1.8, marker="o", ms=3,
                        label=f"{NAME[r]}: 합계")
                ax.plot(ep_, [x["mid_own_mb"] for x in lp[r]], color=col_, lw=1.2, ls="--", label=f"{NAME[r]}: 저장 상태의 배열만")
        ax.set_xlabel("에피소드", fontsize=8.5, color=MUTED); ax.set_ylabel("MB (환경 하나)", fontsize=8.5, color=MUTED)
        ax.set_ylim(0, 45)
        ax.set_title("메모리 (환경 하나): 저장 상태 최근 4개의 배열 +\n그 상태들만 붙잡은 시나리오 (하나 17 MB)", loc="left", fontsize=10.5, color=INK)
        ax.legend(fontsize=8, frameon=False, loc="upper center", ncol=2)
    for a_ in fig.axes:
        if a_.get_label() == "<colorbar>":
            continue
        a_.set_facecolor(BG); a_.tick_params(colors=MUTED, labelsize=8)
        for s_ in a_.spines.values():
            s_.set_color(GRID)
    fig.suptitle("비행 중 시작 (학습 시작 단계 4): 이륙 비행이 10~50분에 상태를 저장 → 다음 에피소드의 30%가 그 상태에서 다섯 가지 시작점 중 하나로 시작 "
                 "(새 시나리오 저장분, 120 km 지도, R12가 대신 비행)", x=0.01, ha="left", fontsize=13, color=INK)
    fig.tight_layout()
    fig.savefig(OUT/"midstart_train_check.png", dpi=80, facecolor=BG)
    # table
    L = ["# 비행 중 시작 (단계 4) 확인", "", "| 지도 | 유형 | 저장 시각 (분) | 저장 뒤 걸음 | 위치 차 (m) | 보상 차 | 관측 | 복원 (ms) | 한 걸음 (ms) |", "|---|---|---|---|---|---|---|---|---|"]
    for r, (res, d) in ex.items():
        L.append(f"| {NAME[r]} | {TN.get(res['typ'], res['typ'])} | {res['save_min']:.1f} | {res['steps_after']} / {res['steps_kept']} | {res['max_pos_m']:.0f} | "
                 f"{res['max_reward']:.0g} | {'같음' if res['same_obs'] else '다름'} | {res['restore_ms']:.0f} | {1000*res['sec_per_step']:.0f} |")
    if lp:
        L += ["", "| 시작 | 에피소드 | 남은 확률 (평균) | 가장 가까운 입자까지 km (중앙값) | R12가 찾은 POS (평균) | 길이 (걸음, 평균) |", "|---|---|---|---|---|---|"]
        for s, lab1 in zip(names, lab_):
            xs = [x for x in rows if x["start"] == s]
            if xs:
                dist = [x["info"].get("dist_km") for x in xs if x["info"].get("dist_km") is not None]
                L.append(f"| {lab1} | {len(xs)} | {np.mean([x['remaining_start'] for x in xs]):.3f} | {np.median(dist) if dist else float('nan'):.1f} | "
                         f"{np.mean([x['pos'] for x in xs]):.3f} | {np.mean([x['steps'] for x in xs]):.0f} |")
    (OUT/"midstart_train_check.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=("exact", "loop", "figure"))
    ap.add_argument("region", nargs="?", default="amboy120")
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--seed", type=int, default=5)
    a = ap.parse_args()
    if a.what == "exact":
        exact(a.region, a.seed)
    elif a.what == "loop":
        loop(a.region, a.episodes, a.seed)
    else:
        figure()


if __name__ == "__main__":
    main()

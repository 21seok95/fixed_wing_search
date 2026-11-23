"""Checks before training A on Chungju 120 km (user 2026-11-18, docs/scenario_redesign.md "학습 A: 충주 120 km 도로형, 이륙만, 이륙 거리 단계"):
launches only, the launch-distance curriculum (10 km, then 10-20, then 10-40 km; the trainer moves on when the policy reaches 0.9 x SAROPS on
the stage's Jirisan cases v2j1 / v2j2, at most 1.5M steps per stage), the road type of the mountain terrain, and the Jirisan validation cases
(v2j). No training here.

- launches: the training env on chungju120 with the run's settings; for each launch-distance stage the ring is set as the trainer does
  (``set_ring``), then ``--n`` episodes are started: start kind, target type, launch distance, launch point (on land closed to a launch?)
- pool: the Chungju 120 km scenarios (rl_pool_v2): count, and the target model settings of a sample (speed spread factor = fast_std / 10,
  heading spread, detours, goal sector, shortcuts, dead ends, the exit rule), with the file times
- cases: the 3 Jirisan validation cases (v2j): LKP, launch, particles at 30 / 60 min, share off the map
- stages: the stage cases v2j1 / v2j2 (the v2j scenarios launched 10 km / 10-20 km away): same LKP and particles as v2j?
- figure: report/launch_ring_check.png and .md, with the SAROPS references of the Jirisan sets; ``--smoke`` a short training run's
  launch-distance stages (state.json and its tb), ``--smoke-old`` an earlier one (table only), ``--regress`` the R12 flights with the new
  settings off against an earlier record (terrain_memory flights)

    python -m rl_search.scripts.rl_eval.launch_ring_check launches --n 100
    python -m rl_search.scripts.rl_eval.launch_ring_check pool
    python -m rl_search.scripts.rl_eval.launch_ring_check cases
    python -m rl_search.scripts.rl_eval.launch_ring_check stages
    python -m rl_search.scripts.rl_eval.launch_ring_check figure --smoke <run dir> --regress maps7_off ring_off
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

OUT = Path("rl_search/data/real/amboy/rl_eval/report")
WORK = OUT/"launch_ring_check"
REGION = "chungju120"
RINGS_KM = ((10.0, 10.0), (10.0, 20.0), (10.0, 40.0))
STAGE_SPLITS = ("v2j1", "v2j2", "v2j")              # Jirisan cases launched at each stage's distance (the last = the validation set)
REF_KEY = "realtime_t5_d3z5l15_s75_o150_i0.75_a0_w1.5"   # the SAROPS reference set of run A's detection and score (R13's)
# the run's settings that shape the scenarios and starts (the observation settings are those of run A: 7 maps, no visibility, from the launch)
SETTINGS = dict(obs_r10=True, no_area=True, a_weight=0.0, maps7=True, no_vis_obs=True, launch_policy=True, scen_v2=True, types="road",
                midflight=0.0)
COLORS = ("#2a6fdb", "#e07b39", "#3a9a5b")          # stage 1 / 2 / 3, as the other check figures
STEP = 8                                            # background maps: every 8th 25 m cell (200 m)


def _closed_layers(terr) -> dict:
    from rl_search.rl_eval_set import _launch_bad
    out = dict(forest=terr.forest, water=terr.water, city=terr.city, steep15=terr.steep(15.0))
    if getattr(terr, "military", None) is not None:
        out["military"] = terr.military
    out["closed"] = _launch_bad(terr)
    return out


def _cell(terr, x, y):
    n = len(terr.axis)
    return tuple(int(np.clip(np.round((v - terr.axis[0])/terr.cell), 0, n - 1)) for v in (y, x))


def launches(n: int):
    from rl_search.rl_real import RealRLConfig, RealRLEnv, pool_dir_multi
    WORK.mkdir(parents=True, exist_ok=True)
    cfg = RealRLConfig(region=REGION, stage=3, settings=dict(SETTINGS), multi=True, pool=pool_dir_multi(SETTINGS, REGION), reuse=2)
    env = RealRLEnv(cfg, seed=7)
    terr = env.terr
    layers = _closed_layers(terr)
    rows = []
    t0 = time.perf_counter()
    for si, (lo, hi) in enumerate(RINGS_KM):
        env.ring_m = (lo*1000.0, hi*1000.0)                 # what the trainer's set_ring does on every worker
        for _ in range(n):
            env.reset()
            e = env.env
            lx, ly = float(e.launch_track[0][1]), float(e.launch_track[0][2])
            lkp = np.asarray(e.scen.lkp, float)
            ij = _cell(terr, lx, ly)
            rows.append(dict(stage=si, start=env.start_kind, typ=env._ep_typ, launch_km=float(env.start_info.get("launch_dist_m", np.nan))/1000,
                             launch=[lx, ly], lkp=[float(lkp[0]), float(lkp[1])], seed=int(e.scen.seed),
                             dist_check_km=float(np.hypot(lx - lkp[0], ly - lkp[1]))/1000,
                             flags={k: bool(v[ij]) for k, v in layers.items()}, obs_keys=sorted(env.observation().keys()) if not rows else None))
        print(f"stage {si} ({lo:g}-{hi:g} km): {n} launches, {time.perf_counter() - t0:.0f} s", flush=True)
    bg = dict(closed=layers["closed"][::STEP, ::STEP], forest=terr.forest[::STEP, ::STEP], axis=terr.axis[::STEP])
    np.savez_compressed(WORK/"chungju_bg.npz", **bg)
    (WORK/"launches.json").write_text(json.dumps(rows))
    share = {k: float(np.mean(v)) for k, v in layers.items()}
    (WORK/"chungju_land.json").write_text(json.dumps(share))
    print("map shares:", {k: round(v, 3) for k, v in share.items()})


def pool(sample_every: int = 40):
    import pickle
    from rl_search.rl_real import pool_dir_multi
    d = Path(pool_dir_multi(SETTINGS, REGION))
    files = sorted(d.glob("road_s*.pkl"), key=lambda p: int(p.stem.split("_s")[1]))
    pick = files[:60] + files[60::sample_every]
    keys = ("fast_std_kmh", "slow_std_kmh", "off_std_kmh", "heading_std_deg", "reorient_std_deg_min", "p_detour", "goal_sector_deg",
            "detour_keep_tier", "deadend_offroad", "shortcut_m", "exit_straight", "exit_trend_min", "p_leave_fast_min", "speed_clip_sd")
    rows = []
    for p in pick:
        with open(p, "rb") as f:
            dd = pickle.load(f)
        vc = dd.get("vc") or {}
        rows.append(dict(seed=int(p.stem.split("_s")[1]), mtime=os.path.getmtime(p), type=dd.get("type"),
                         **{k: (vc.get(k) if not isinstance(vc.get(k), np.generic) else vc.get(k).item()) for k in keys}))
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK/"pool.json").write_text(json.dumps(dict(n_files=len(files), rows=rows), default=float))
    f = np.array([r["fast_std_kmh"]/10.0 for r in rows])
    print(f"{len(files)} files; sample {len(rows)}: spread factor {f.min():.2f}-{f.max():.2f}, above 1.0: {int((f > 1.0 + 1e-9).sum())} "
          f"(of the first 60: {int((f[:60] > 1.0 + 1e-9).sum())})")


def cases():
    from rl_search.rl_eval_set import MULTI_N
    from rl_search.rl_real import preload
    from rl_search.rl_real_eval import _case
    WORK.mkdir(parents=True, exist_ok=True)
    out = []
    for k in range(MULTI_N["v2j"]):
        t0 = time.perf_counter()
        case, scen, vres, hold = _case("v2j", 3, k)
        T = np.asarray(scen.track)
        dt = float(scen.save_dt_s)
        snaps = {}
        for m in (0, 30, 60):
            i = min(int(round(m*60/dt)), len(T) - 1)
            P = T[i]
            snaps[m] = P[:: max(len(P)//1500, 1)].tolist()
        out.append(dict(k=k, label=case.label, typ=case.typ, lkp=case.lkp, start=case.start, info=case.info,
                        vc={k2: (float(v) if isinstance(v, (int, float, np.generic)) and not isinstance(v, bool) else v) for k2, v in case.vc.items()},
                        snaps=snaps, sec=time.perf_counter() - t0))
        print(f"v2j {k}: {case.label}, launch {case.info['launch_dist_m']/1000:.1f} km, off the map at 60 min {case.info['out_share']:.1%}, "
              f"{time.perf_counter() - t0:.0f} s", flush=True)
    terr = preload("jirisan120")["terr"]
    layers = _closed_layers(terr)
    np.savez_compressed(WORK/"jirisan_bg.npz", closed=layers["closed"][::STEP, ::STEP], forest=terr.forest[::STEP, ::STEP], axis=terr.axis[::STEP])
    (WORK/"cases.json").write_text(json.dumps(out, default=float))


def stages():
    """The launch-distance stages' validation cases (v2j1 10 km, v2j2 10-20 km): the v2j scenarios with another launch. Same LKP, heading and
    particles (planner and hold-out) as v2j? Launch distance of each."""
    from rl_search.rl_eval_set import MULTI_N
    from rl_search.rl_real_eval import _case
    out = []
    for k in range(MULTI_N["v2j"]):
        c0, s0, _, h0 = _case("v2j", 3, k)
        for sp in ("v2j1", "v2j2"):
            c, s, _, h = _case(sp, 3, k)
            out.append(dict(split=sp, k=k, label=c.label, launch_km=c.info["launch_dist_m"]/1000, start=c.start, lkp=c.lkp,
                            same_lkp=bool(np.allclose(c.lkp, c0.lkp)), same_heading=bool(c.heading == c0.heading),
                            same_particles=bool(np.array_equal(np.asarray(s.track), np.asarray(s0.track))),
                            same_holdout=bool(np.array_equal(np.asarray(h.track), np.asarray(h0.track)))))
            print(f"{sp} {k}: launch {out[-1]['launch_km']:.1f} km, same LKP {out[-1]['same_lkp']}, same particles {out[-1]['same_particles']}, "
                  f"same hold-out {out[-1]['same_holdout']}", flush=True)
    (WORK/"stages.json").write_text(json.dumps(out, default=float))


def _refs() -> dict:
    """SAROPS references of the Jirisan sets (run A's reference key): {split: {k: dict(total, pos, start, lkp)}}."""
    from rl_search.scripts.rl_eval.sarops_ref import ref_path
    out = {}
    for sp in STAGE_SPLITS:
        p = ref_path(sp, REF_KEY)
        if p.exists():
            out[sp] = {int(r["case"]["k"]): dict(total=float(r["holdout"]["total"]), pos=float(r["holdout"]["pos"]),
                                                 start=r["case"]["start"], lkp=r["case"]["lkp"]) for r in json.loads(p.read_text())}
    return out


def _smoke(run: Path) -> dict | None:
    """Launch-distance stages of a short training run: state.json (blocks, evaluations, switches) and the tb scalar episode/launch_km."""
    if run is None or not (run/"state.json").exists():
        return None
    st = json.loads((run/"state.json").read_text())
    out = dict(ring=st.get("ring"), tb={})
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        for ev in sorted((run/"tb").rglob("events.*")):
            acc = EventAccumulator(str(ev.parent))
            acc.Reload()
            for tag in ("ring/stage", "episode/launch_km", "episode/POS"):
                if tag in acc.Tags().get("scalars", []):
                    out["tb"][tag] = [(s.step, s.value) for s in acc.Scalars(tag)]
    except Exception as exc:                                   # the figure still shows state.json
        out["tb_error"] = repr(exc)
    return out


def _regress(before: str, after: str) -> list | None:
    d = OUT/"terrain_memory"
    pb, pa = d/f"flights_{before}.json", d/f"flights_{after}.json"
    if not (pb.exists() and pa.exists()):
        return None
    fb, fa = json.loads(pb.read_text()), json.loads(pa.read_text())
    return [dict(case=c, steps=fb[c]["steps"], pos_b=fb[c]["pos"][-1], pos_a=fa[c]["pos"][-1],
                 same=fb[c]["pos"] == fa[c]["pos"] and fb[c]["obs_hash"] == fa[c]["obs_hash"]) for c in fb]


def _smoke_md(sm: dict, title: str) -> list:
    R = sm["ring"]
    L = ["", f"## {title}", "", "| 블록 끝 스텝 | 단계 | 이륙 비행 수 | 이륙 POS 평균 |", "|---|---|---|---|"]
    L += [f"| {b['step']:,} | {b['stage'] + 1} | {b['n']} | {b['pos']:.4f} |" for b in R["blocks"]]
    if R.get("evals"):
        L += ["", "| 평가 스텝 | 단계 | 사례 | RL | SAROPS | 비율 |", "|---|---|---|---|---|---|"]
        L += [f"| {e['step']:,} | {e['stage'] + 1} | {e['split']} | {np.mean(e['rl']):.3f} | {np.mean(e['sarops']):.3f} | {e['ratio']:.3f} |"
              for e in R["evals"]]
    L += [""] + [f"- {w['step']:,} 스텝: {w['from_stage'] + 1} → {w['to_stage'] + 1}단계 ({w['why']})" for w in R["switches"]]
    return L


def figure(smoke: Path | None, regress: tuple | None, smoke_old: Path | None = None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    plt.rcParams["font.family"] = "AppleGothic"
    plt.rcParams["axes.unicode_minus"] = False
    rows = json.loads((WORK/"launches.json").read_text())
    land = json.loads((WORK/"chungju_land.json").read_text())
    pl = json.loads((WORK/"pool.json").read_text())
    cs = json.loads((WORK/"cases.json").read_text())
    stg = json.loads((WORK/"stages.json").read_text()) if (WORK/"stages.json").exists() else []
    refs = _refs()
    sm, sm_old = _smoke(smoke), _smoke(smoke_old)
    rg = _regress(*regress) if regress else None
    fig = plt.figure(figsize=(25, 12))
    gs = fig.add_gridspec(2, 4, hspace=0.34, wspace=0.24)
    names = [f"{i + 1}단계 {lo:g} km" if lo == hi else f"{i + 1}단계 {lo:g}~{hi:g} km" for i, (lo, hi) in enumerate(RINGS_KM)]

    ax = fig.add_subplot(gs[0, 0])                              # (a) launch distance per stage
    bins = np.arange(9.5, 41.0, 1.0)
    for si in range(3):
        d = [r["launch_km"] for r in rows if r["stage"] == si]
        ax.hist(d, bins=bins, color=COLORS[si], alpha=.85, label=f"{names[si]} (이륙 {len(d)}번)")
    ax.set_xlabel("LKP에서 이륙점까지 (km)")
    ax.set_ylabel("이륙 수")
    ax.set_title("(가) 단계마다 이륙 거리 (충주, 학습 환경에서 이륙만 뽑음)", fontsize=10)
    ax.legend(fontsize=8, frameon=False)

    ax = fig.add_subplot(gs[0, 1])                              # (b) launch points around the LKP
    for si in (0, 1, 2):
        P = np.array([np.subtract(r["launch"], r["lkp"]) for r in rows if r["stage"] == si])/1000
        ax.scatter(P[:, 0], P[:, 1], s=9, color=COLORS[si], label=names[si], alpha=.85, linewidths=0, zorder=3 - si)
    for rad in (10, 20, 40):
        ax.add_patch(plt.Circle((0, 0), rad, fill=False, color="#8a8f98", lw=.7, ls="--"))
    ax.plot(0, 0, "*", color="#d9480f", ms=12, label="LKP")
    ax.set_aspect("equal")
    ax.set_xlim(-43, 43)
    ax.set_ylim(-43, 43)
    ax.set_xlabel("동쪽 (km)")
    ax.set_ylabel("북쪽 (km)")
    ax.set_title("(나) LKP를 가운데 두고 다시 그린 이륙점 (점선 10 / 20 / 40 km)\nLKP 자체는 시나리오마다 다름 → (다)", fontsize=10)
    ax.legend(fontsize=8, frameon=False, loc="upper right")

    def bg(ax, path, title):
        b = np.load(path)
        a = b["axis"]/1000
        ax.imshow(b["closed"].astype(np.uint8), origin="lower", extent=(a[0], a[-1], a[0], a[-1]), cmap=ListedColormap(["#ffffff", "#c9cdd2"]),
                  vmin=0, vmax=1, interpolation="nearest")
        ax.set_xlabel("동쪽 (km)")
        ax.set_ylabel("북쪽 (km)")
        ax.set_title(title, fontsize=10)

    ax = fig.add_subplot(gs[0, 2])                              # (c) launch points on the Chungju map
    bg(ax, WORK/"chungju_bg.npz", f"(다) 충주 120 km 실제 위치: 빨간 별 = 그 이륙의 LKP(가운데 30 km 안 도로)\n회색 = 이륙 못 하는 땅(숲·경사 15°+·물·도시·군사, {land['closed']:.0%})")
    for si in (2, 1, 0):
        P = np.array([r["launch"] for r in rows if r["stage"] == si])/1000
        ax.scatter(P[:, 0], P[:, 1], s=8, color=COLORS[si], alpha=.9, linewidths=0)
    L = np.array([r["lkp"] for r in rows])/1000
    ax.scatter(L[:, 0], L[:, 1], marker="*", s=40, color="#d9480f", linewidths=0, label="LKP")
    ax.add_patch(plt.Rectangle((-15, -15), 30, 30, fill=False, color="#d9480f", lw=.8, ls=":"))
    ax.legend(fontsize=8, frameon=False, loc="lower right")

    ax = fig.add_subplot(gs[0, 3])                              # (d) pool: speed spread factor by seed
    pr = pl["rows"]
    s = np.array([r["seed"] for r in pr]) - 14_000_000
    fct = np.array([r["fast_std_kmh"]/10.0 for r in pr])
    first = s < 60
    ax.scatter(s[~first], fct[~first], s=12, color="#2a6fdb", label=f"나머지 표본 ({int((~first).sum())}개)", linewidths=0)
    ax.scatter(s[first], fct[first], s=12, color="#e07b39", label="먼저 만든 60개 (씨앗 0~59)", linewidths=0)
    ax.axhspan(0.5, 1.0, color="#3a9a5b", alpha=.12, label="정한 범위 0.5~1.0")
    ax.set_xscale("symlog", linthresh=60)
    ax.set_xlabel("씨앗 - 14,000,000 (만든 차례)")
    ax.set_ylabel("속도 퍼짐 배율 (포장 표준편차 / 10 km/h)")
    ax.set_title(f"(라) 충주 저장분 {pl['n_files']:,}개 중 표본 {len(pr)}개의 속도 퍼짐 배율", fontsize=10)
    ax.legend(fontsize=8, frameon=False)

    ax = fig.add_subplot(gs[1, 0])                              # (e) Jirisan cases and the three launches of each
    bg(ax, WORK/"jirisan_bg.npz", "(마) 지리산 사례 3개: 입자 30 · 60분(회색), LKP(별)\n이륙 ▲ 10 km(1단계용) ■ 10~20 km(2단계용) ● 10~40 km(검증)")
    for c in cs:
        for m, a in ((30, .25), (60, .45)):
            P = np.array(c["snaps"][str(m)])/1000
            ax.scatter(P[:, 0], P[:, 1], s=1.5, color="#5b6068", alpha=a, linewidths=0)
        lk = np.array(c["lkp"])/1000
        starts = {"v2j": np.array(c["start"][:2])/1000} | {r["split"]: np.array(r["start"][:2])/1000 for r in stg if r["k"] == c["k"]}
        for si, (sp, mk) in enumerate(zip(STAGE_SPLITS, ("^", "s", "o"))):
            if sp in starts:
                st = starts[sp]
                ax.plot([st[0], lk[0]], [st[1], lk[1]], color=COLORS[si], lw=.8, ls=":")
                ax.plot(*st, mk, color=COLORS[si], ms=7, mec="white", mew=.7)
        ax.plot(*lk, "*", color="#d9480f", ms=12, mec="white", mew=.6)
        ax.annotate(f"사례 {c['k'] + 1}", starts["v2j"], textcoords="offset points", xytext=(7, -3), fontsize=9, color="#222",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=.8))      # at its 10-40 km launch: the LKPs are close

    ax = fig.add_subplot(gs[1, 1])                              # (f) SAROPS by launch distance
    width = .26
    for si, sp in enumerate(STAGE_SPLITS):
        if sp not in refs:
            continue
        ks = sorted(refs[sp])
        v = [refs[sp][k]["total"] for k in ks]
        x = np.arange(len(ks)) + (si - 1)*width
        ax.bar(x, v, width*.92, color=COLORS[si], label=f"{names[si].split(' ', 1)[1]} 이륙 ({sp}, 평균 {np.mean(v):.3f})")
        for xi, vi in zip(x, v):
            ax.annotate(f"{vi:.2f}", (xi, vi), textcoords="offset points", xytext=(0, 2), ha="center", fontsize=7, color="#333")
    ax.set_ylim(0, max(v["total"] for sp in refs for v in refs[sp].values())*1.45)    # room for the legend above the bars
    ax.set_xticks(range(3))
    ax.set_xticklabels([f"사례 {k + 1}" for k in range(3)])
    ax.set_ylabel("SAROPS 점수 (1.5 × POS, 별도 입자로 채점)")
    ax.set_title("(바) 같은 시나리오를 이륙 거리만 바꿔 SAROPS로 날린 점수\n(단계 넘김 기준: RL이 이것의 0.9배, 최근 3번 평균)", fontsize=10)
    ax.legend(fontsize=8, frameon=False, loc="upper left")

    ax = fig.add_subplot(gs[1, 2])                              # (g) the smoke run's stages
    ax2 = fig.add_subplot(gs[1, 3])                             # (h) its stage evaluations
    if sm and sm.get("ring"):
        R = sm["ring"]
        tb = sm["tb"]
        end = max([b["step"] for b in R["blocks"]] + [e["step"] for e in R.get("evals", [])] + [1])
        edges = [0] + [w["step"] for w in R["switches"]] + [end]
        for a_ in (ax, ax2):
            for i in range(len(edges) - 1):
                a_.axvspan(edges[i], edges[i + 1], color=COLORS[min(i, 2)], alpha=.12)
            for w in R["switches"]:
                a_.axvline(w["step"], color="#5b6068", lw=.8, ls="--")
        for i in range(len(edges) - 1):
            lo, hi = RINGS_KM[min(i, 2)]
            ax.hlines([lo, hi], edges[i], edges[i + 1], color=COLORS[min(i, 2)], lw=3, alpha=.6)
        if "episode/launch_km" in tb:
            x, y = zip(*tb["episode/launch_km"])
            ax.plot(x, y, color="#222", lw=1.2, ls="--", label="끝난 비행의 이륙 거리 평균 (최근 100번)", zorder=5)
        for w in R["switches"]:
            ax.annotate(f"{w['from_stage'] + 1}→{w['to_stage'] + 1}단계", (w["step"], 41.5), fontsize=8, ha="center", color="#333")
        ax.set_ylim(0, 44)
        ax.set_xlabel("학습 스텝")
        ax.set_ylabel("이륙 거리 (km)")
        ax.legend(fontsize=8, frameon=False, loc="center right")
        ev = R.get("evals", [])
        for si in range(3):
            e = [x for x in ev if x["stage"] == si]
            if e:
                ax2.plot([x["step"] for x in e], [x["ratio"] for x in e], "o", color=COLORS[si], ms=5, label=f"{names[si]}: {e[0]['split']} 평가")
                m3 = [np.mean([y["ratio"] for y in e[max(0, j - 2):j + 1]]) for j in range(len(e))]
                ax2.plot([x["step"] for x in e], m3, "-", color=COLORS[si], lw=1.2)
        ax2.axhline(0.9, color="#d9480f", lw=1, ls=":", label="실제 학습의 넘김 기준 0.9")
        ax2.set_xlabel("학습 스텝")
        ax2.set_ylabel("RL / SAROPS (그 단계 사례 3개)")
        ax2.legend(fontsize=8, frameon=False, loc="upper right")
        ax2.set_ylim(0, 1.05)
    ax.set_title("(사) 짧은 시험 학습(맥, 환경 2개): 단계와 이륙 거리\n(색 띠 = 단계, 굵은 선 = 그 단계 거리 범위, 점선 = 넘긴 때)", fontsize=10)
    ax2.set_title("(아) 같은 시험의 단계 평가: 점 = 평가 한 번, 선 = 최근 3번 평균\n(길 확인용으로 넘김 기준을 0으로 낮춤 → 3번 평가 뒤 넘김)", fontsize=10)
    fig.suptitle("학습 A 준비: 충주 120 km 도로형 · 이륙만 · 이륙 거리 단계 10 → 10~20 → 10~40 km (SAROPS의 0.9배면 넘김, 단계당 최대 150만) · 지리산 검증",
                 fontsize=13)
    fig.savefig(OUT/"launch_ring_check.png", dpi=105, bbox_inches="tight")

    L = ["# 학습 A 준비 확인 (사용자 11-18)", "", "그림 `report/launch_ring_check.png`. 학습은 하지 않음(시험 학습은 단계 바꾸기 확인용, 환경 2개).", "",
         "## 이륙 (충주 120 km, 학습 환경 그대로)", "", "| 단계 | 이륙 수 | 거리 최소 / 중앙 / 최대 (km) | 시작 종류 | 표적 유형 | 이륙 못 하는 땅 위 |", "|---|---|---|---|---|---|"]
    for si in range(3):
        rr = [r for r in rows if r["stage"] == si]
        d = np.array([r["launch_km"] for r in rr])
        kinds = {k: sum(r["start"] == k for r in rr) for k in sorted({r["start"] for r in rr})}
        typs = {k: sum(r["typ"] == k for r in rr) for k in sorted({r["typ"] for r in rr})}
        bad = sum(r["flags"]["closed"] for r in rr)
        L.append(f"| {names[si]} | {len(rr)} | {d.min():.2f} / {np.median(d):.2f} / {d.max():.2f} | {kinds} | {typs} | {bad} |")
    gap = max(abs(r["launch_km"] - r["dist_check_km"]) for r in rows)
    lk = np.array([r["lkp"] for r in rows])/1000
    L += ["", f"- LKP(시나리오마다 다름): 동쪽 {lk[:, 0].min():.1f}~{lk[:, 0].max():.1f} km, 북쪽 {lk[:, 1].min():.1f}~{lk[:, 1].max():.1f} km(지도 가운데 기준). "
          "이륙 순간 기체는 LKP를 향함 → 무인기 중심 관측에서는 1단계 이륙 때 입자가 늘 정면 10 km 한 점.",
          f"- 이륙점과 LKP 사이를 다시 잰 거리와 기록 거리의 차이 최대 {gap*1000:.2f} m.",
          f"- 지도에서 이륙 못 하는 땅: 숲 {land['forest']:.1%}, 경사 15° 이상 {land['steep15']:.1%}, 물 {land['water']:.1%}, 도시 {land['city']:.1%}"
          + (f", 군사 {land['military']:.1%}" if "military" in land else "") + f" → 합 {land['closed']:.1%}.",
          f"- 관측 지도: {', '.join(k for k in rows[0]['obs_keys'] if k != 'vec')} + 벡터.", "",
          "## 충주 저장분", "", f"파일 {pl['n_files']:,}개, 표본 {len(pr)}개(먼저 만든 60개 + 그 뒤 40개마다 하나).", "",
          "| 무엇 | 먼저 만든 60개 | 나머지 표본 |", "|---|---|---|"]
    for key, fmt in (("fast_std_kmh", "{:.2f}~{:.2f}"), ("heading_std_deg", "{:.0f}~{:.0f}"), ("reorient_std_deg_min", "{:.2f}~{:.2f}"),
                     ("p_detour", "{:.3f}~{:.3f}"), ("goal_sector_deg", "{:.0f}~{:.0f}"), ("exit_trend_min", "{:g}~{:g}"), ("shortcut_m", "{:g}~{:g}")):
        fa = np.array([float(r[key]) for r, f0 in zip(pr, first) if f0 and r.get(key) is not None])
        fb = np.array([float(r[key]) for r, f0 in zip(pr, first) if not f0 and r.get(key) is not None])
        L.append(f"| {key} | {fmt.format(fa.min(), fa.max()) if len(fa) else '없음'} | {fmt.format(fb.min(), fb.max()) if len(fb) else '없음'} |")
    for key in ("detour_keep_tier", "deadend_offroad", "exit_straight"):
        L.append(f"| {key} | {sorted({str(r[key]) for r, f0 in zip(pr, first) if f0})} | {sorted({str(r[key]) for r, f0 in zip(pr, first) if not f0})} |")
    L += ["", "## 지리산 사례 (v2j 검증, v2j1 / v2j2 단계 넘김용)", "",
          "| 사례 | 10~40 km 이륙 (v2j) | 10 km 이륙 (v2j1) | 10~20 km 이륙 (v2j2) | 60분까지 지도 밖 | 방향 퍼짐 (°) | 우회 | 속도 퍼짐 배율 |",
          "|---|---|---|---|---|---|---|---|"]
    for c in cs:
        k = c["k"]

        def cell(sp, km):
            r = refs.get(sp, {}).get(k)
            return f"{km:.1f} km, SAROPS {r['total']:.3f}" if r else f"{km:.1f} km"
        km = {r["split"]: r["launch_km"] for r in stg if r["k"] == k}
        L.append(f"| {k + 1} | {cell('v2j', c['info']['launch_dist_m']/1000)} | {cell('v2j1', km.get('v2j1', float('nan')))} | "
                 f"{cell('v2j2', km.get('v2j2', float('nan')))} | {c['info']['out_share']:.1%} | {c['vc'].get('heading_std_deg', float('nan')):.0f} | "
                 f"{c['vc'].get('p_detour', float('nan')):.3f} | {c['vc'].get('fast_std_kmh', float('nan'))/10:.2f} |")
    if stg:
        ok = all(r["same_lkp"] and r["same_heading"] and r["same_particles"] and r["same_holdout"] for r in stg)
        L += ["", f"- v2j1 / v2j2가 v2j와 LKP, 도주 방향, 입자(계획용·채점용)가 모두 같은가: {'예' if ok else '아니요'} ({len(stg)}개 사례)."]
    for sp in STAGE_SPLITS:
        if sp in refs:
            v = [refs[sp][k]["total"] for k in sorted(refs[sp])]
            p = [refs[sp][k]["pos"] for k in sorted(refs[sp])]
            L.append(f"- SAROPS {sp}: 점수 평균 {np.mean(v):.3f} (POS 평균 {np.mean(p):.3f}), 사례별 {', '.join(f'{x:.3f}' for x in v)}.")
    if sm and sm.get("ring"):
        L += _smoke_md(sm, "짧은 시험 학습: SAROPS 기준 단계 넘김 (길 확인용으로 넘김 기준 0)")
    if sm_old and sm_old.get("ring"):
        L += _smoke_md(sm_old, "앞선 시험 학습: 이륙 POS가 멈추면 넘김 (쓰지 않기로 함)")
    if rg:
        L += ["", "## 새 설정을 끈 R12 비행 (예전과 같은가)", ""]
        L += [f"- {r['case']}: 결정 {r['steps']}번, 최종 POS {r['pos_b']:.4f} / {r['pos_a']:.4f}, 매 걸음 POS와 관측 {'같음' if r['same'] else '다름'}" for r in rg]
    (OUT/"launch_ring_check.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("launches", "pool", "cases", "stages", "figure"))
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--smoke", type=Path, default=None, help="a short training run's folder (state.json, tb)")
    ap.add_argument("--smoke-old", type=Path, default=None, help="an earlier short run (table only)")
    ap.add_argument("--regress", nargs=2, default=None, help="terrain_memory flights labels: before after")
    a = ap.parse_args()
    if a.what == "launches":
        launches(a.n)
    elif a.what == "pool":
        pool()
    elif a.what == "cases":
        cases()
    elif a.what == "stages":
        stages()
    else:
        figure(a.smoke, tuple(a.regress) if a.regress else None, a.smoke_old)


if __name__ == "__main__":
    main()

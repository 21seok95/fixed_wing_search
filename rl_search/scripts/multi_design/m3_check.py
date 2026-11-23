"""Checks before the two-UAV run (user 2026-11-24: "확인사항들은 확인하고 넘어가면 좋겠습니다"; design m3 = transit / search collision rules,
rl_multi.DESIGNS): R16 flown by two UAVs, each on its own inputs (``solo_view``: the teammate channels read "no teammate", as R16 trained),
on the three Jirisan cases, under m3 and under m2's rules. Recorded per step: modes, pair geometry, filter cuts, violations, reward parts;
the penalty is recomputed offline for both danger shapes ("step", "linear"), since R16 does not react to it. Two moments are kept for the
channel figure: three minutes after the second launch (both in transit, flying in a column) and the first step with one UAV searching and
the pair 600-1000 m apart.

    python -m rl_search.scripts.multi_design.m3_check fly --case 0 --design m3      (one flight; run the six in parallel)
    python -m rl_search.scripts.multi_design.m3_check report
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
import pickle
import warnings
from pathlib import Path

import numpy as np

from rl_search.scripts.multi_design.ma_eval import OUT, _gaps

MODEL = "rl_search/data/rl_real/r16_chungju/ckpt/model_6000000.zip"
WORK = OUT/"m3_check"
SHAPES = ("step", "linear")


def _channels(env, i: int) -> dict:
    """UAV i's danger channel on the three small maps under m2's 250 m disk and m3's graded shapes, from the same teammate fans."""
    from rl_search.mate_maps import danger_map, danger_map_graded
    from rl_search.rl_multi import danger_level, SMALL_SIDE
    x, y, yaw = env.uavs[i].env.plant.pose
    px = env.uavs[i].cfg.map_px
    allf = env._fans()
    fans = [allf[j] for j in np.flatnonzero(env.active) if j != i and j in allf]
    out = {}
    for name, side, lead in env.maps:
        if side > SMALL_SIDE:
            continue
        out[name] = dict(side=side, m2=danger_map(x, y, yaw, fans, side, float(lead), 250.0, px),
                         **{s: danger_map_graded(x, y, yaw, fans, side, float(lead), lambda d, s=s: danger_level(d, s), 1000.0, px) for s in SHAPES})
    mates = [j for j in np.flatnonzero(env.active) if j != i]
    rel = []
    for j in mates:
        d = np.asarray(env.uavs[j].env.plant.pose[:2]) - np.asarray([x, y])
        rel.append((float(d[0]*np.cos(yaw) + d[1]*np.sin(yaw)), float(d[0]*np.sin(yaw) - d[1]*np.cos(yaw))))   # (forward, right)
    return dict(maps=out, mates=rel, t=float(env.T), searching=env.searching.tolist() if hasattr(env, "searching") else [])


def fly(case_k: int, design: str, model_path: str = MODEL):
    warnings.filterwarnings("ignore")
    import torch
    from stable_baselines3 import PPO
    from rl_search.rl_multi import MultiRealEnv
    from rl_search.rl_real import RealRLConfig
    from rl_search.rl_real_eval import _case
    from rl_search.scripts.rl_eval.signal_check import settings_of
    torch.set_num_threads(1)
    s = settings_of(json.loads((Path(model_path).parent.parent/"config.json").read_text()))
    model = PPO.load(model_path, device="cpu")
    case, scen, _, _ = _case("v2j", 3, case_k)
    env = MultiRealEnv(2, RealRLConfig(region="jirisan120", settings=s, mask_cost=0.1), seed=case_k, design=design)
    env.solo_view = True
    obs = env.reset(options=case.options(scen), gaps=_gaps(case_k, 2, "m2"))
    rows, snaps = [], {}
    while not env.done:
        batch = {k: np.stack([o[k] for o in obs]) for k in obs[0]}
        a, _ = model.predict(batch, deterministic=True)
        obs, rew, _, info = env.step([np.atleast_1d(x) for x in np.ravel(a)])
        act = np.asarray(info["active"], bool)
        d_now = float(np.hypot(*(env.uavs[0].env.plant.pose[:2] - env.uavs[1].env.plant.pose[:2])))
        searching = env.searching.copy() if design == "m3" else np.zeros(2, bool)
        rows.append(dict(t=float(env.T), active=act.tolist(), searching=searching.tolist(), cut=np.asarray(info["cut"]).tolist(),
                         viol=np.asarray(info["violation"]).tolist(), parts=np.asarray(info["parts"]).tolist(), d_now=d_now,
                         dmin=float(info["dmin"]), pairs=info.get("pairs", []), found=float(env.found)))
        both = env.active.all()
        if design == "m3" and both:
            if "transit" not in snaps and not searching.any() and env.T - env.launch_t[1] >= 180.0:
                snaps["transit"] = _channels(env, 1)
            if "search" not in snaps and searching.any() and 600.0 <= d_now <= 1000.0:
                snaps["search"] = _channels(env, int(np.flatnonzero(searching)[0]))
    WORK.mkdir(parents=True, exist_ok=True)
    out = dict(case=case_k, design=design, model=model_path, launch_t=env.launch_t.tolist(), found=float(env.found),
               search_t=(env.search_t.tolist() if design == "m3" else None), rows=rows,
               hist=[np.array(h)[::5].tolist() for h in env.hist])
    (WORK/f"{design}_c{case_k}.json").write_text(json.dumps(out))
    if snaps:
        pickle.dump(snaps, open(WORK/f"{design}_c{case_k}_snaps.pkl", "wb"))
    print(f"{design} case {case_k + 1}: team POS {env.found:.3f}, search from "
          f"{[None if not np.isfinite(x) else round((x - l)/60, 1) for x, l in zip(env.search_t, env.launch_t)] if design == 'm3' else '-'} min, "
          f"snapshots {sorted(snaps)}", flush=True)


def _penalties(r: dict, shape: str) -> np.ndarray:
    """Per UAV the m3 penalty of a step recomputed for ``shape`` (transit: m2's formula), from the recorded pair geometry."""
    from rl_search.rl_multi import C_CPA, C_DIST, LOOK_S, danger_level
    out = np.zeros(2)
    for (i, j, d_now, d_min, rdot, t_c, d_c, dh, s_i, s_j) in r["pairs"]:
        for u, srch in ((i, s_i), (j, s_j)):
            if srch:
                out[u] += (C_DIST*float(danger_level(d_now, shape)) if rdot <= 3.0 else 0.0) + \
                    (C_CPA*(1 - t_c/35.0)*float(danger_level(d_c, shape)) if t_c <= 35.0 else 0.0)
            else:
                out[u] += (C_CPA*(1 - t_c/LOOK_S)*(1 - d_c/250.0) if (t_c <= LOOK_S and d_c < 250.0) else 0.0) + \
                    (C_DIST*(1 - d_now/250.0) if (d_now < 250.0 and rdot < 0.0) else 0.0)
    return out


def _stats(d: dict) -> dict:
    rows = d["rows"]
    strict = np.array([any(p[8] or p[9] for p in r["pairs"]) for r in rows]) if d["design"] == "m3" else np.zeros(len(rows), bool)
    both = np.array([all(r["active"]) for r in rows])
    cut = np.array([max(r["cut"]) > 0 for r in rows])
    gain = np.array([sum(r["parts"][0]) + sum(r["parts"][1]) for r in rows])
    out = dict(found=d["found"], cuts_transit=int((cut & ~strict).sum()), cuts_search=int((cut & strict).sum()),
               viol=float(sum(sum(r["viol"]) for r in rows)/2), gain=float(gain.sum()))
    dn = np.array([r["d_now"] for r in rows])
    m = strict & both
    out["search_steps"] = int(m.sum())
    out["in1km"] = float((dn[m] < 1000).mean()) if m.any() else float("nan")
    out["dmin_search"] = float(np.min([r["dmin"] for r, k in zip(rows, m) if k])) if m.any() else float("nan")
    out["dmin_all"] = float(np.nanmin([r["dmin"] for r in rows if np.isfinite(r["dmin"])]))
    if d["design"] == "m3":
        for s in SHAPES:
            pen = np.array([_penalties(r, s).sum() for r in rows])
            out[f"pen_{s}"] = float(pen.sum())
            out[f"share_{s}"] = float(pen.sum()/max(gain.sum(), 1e-9))
        out["search_min"] = [None if x is None or not np.isfinite(x) else round((x - l)/60, 1) for x, l in zip(d["search_t"], d["launch_t"])]
    else:
        pen = np.array([sum(r["parts"][3]) for r in rows])
        out["pen_m2"] = float(pen.sum())
        out["share_m2"] = float(pen.sum()/max(gain.sum(), 1e-9))
    return out


def report():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    plt.rcParams["font.family"] = "AppleGothic"
    plt.rcParams["axes.unicode_minus"] = False
    runs = {(dz, c): json.loads((WORK/f"{dz}_c{c}.json").read_text()) for dz in ("m3", "m2") for c in range(3) if (WORK/f"{dz}_c{c}.json").exists()}
    stats = {key: _stats(d) for key, d in runs.items()}
    print("case design | team POS | search from (min after own launch) | filter cuts transit / search | violations | min dist all / search (m) "
          "| search steps, share within 1 km | penalty / gain: step, linear (m3) or m2")
    for (dz, c), s in sorted(stats.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        pen = (f"{s['share_step']*100:.1f}% , {s['share_linear']*100:.1f}%" if dz == "m3" else f"{s['share_m2']*100:.1f}%")
        print(f"{c + 1} {dz} | {s['found']:.3f} | {s.get('search_min', '-')} | {s['cuts_transit']} / {s['cuts_search']} | {s['viol']:.0f} | "
              f"{s['dmin_all']:.0f} / {s['dmin_search']:.0f} | {s['search_steps']}, {s['in1km']*100:.0f}% | {pen} (gain {s['gain']:.0f})")
    (WORK/"stats.json").write_text(json.dumps({f"{dz}_c{c}": s for (dz, c), s in stats.items()}, indent=1))

    # figure 1: the danger channel at the two moments
    snaps = None
    for c in range(3):
        p = WORK/f"m3_c{c}_snaps.pkl"
        if p.exists():
            sn = pickle.load(open(p, "rb"))
            if "transit" in sn and "search" in sn:
                snaps, snap_case = sn, c
                break
    if snaps:
        moments = [("transit", "이동 중 (두 번째 이륙 3분 뒤, 줄지어 비행)"), ("search", "탐색 중 (한 대가 탐색, 둘 사이 600~1000 m)")]
        kinds = [("m2", "지금 m2: 250 m 원판"), ("step", "제안 m3 step: 500 m 안 1, 0.7→0.3 (1 km)"), ("linear", "m3 linear: 1 - 거리/1 km")]
        fig, ax = plt.subplots(len(kinds), 6, figsize=(24, 12.5))
        for mi, (mom, mtitle) in enumerate(moments):
            sn = snaps[mom]
            for ki, (kind, ktitle) in enumerate(kinds):
                for ci, name in enumerate(("m1600", "m3200", "m6400")):
                    a = ax[ki, mi*3 + ci]
                    mp = sn["maps"][name]
                    h = mp["side"]/2000
                    a.imshow(mp[kind], cmap="magma", vmin=0, vmax=1, extent=(-h, h, -h, h), origin="upper")
                    a.plot(0, 0, marker="^", color="#38b2ac", ms=9)
                    for f, r in sn["mates"]:
                        a.plot(r/1000, f/1000, marker="X", color="#63b3ed", ms=9, mec="k")
                        for rad, ls in ((0.5, "-"), (1.0, "--")):
                            a.add_patch(Circle((r/1000, f/1000), rad, fill=False, color="#63b3ed", ls=ls, lw=1))
                    a.set_xlim(-h, h)
                    a.set_ylim(-h, h)
                    a.set_xticks([])
                    a.set_yticks([])
                    if ki == 0:
                        a.set_title(f"{mtitle if ci == 1 else ''}\n{mp['side']/1000:g} km 지도", fontsize=10)
                    if mi == 0 and ci == 0:
                        a.set_ylabel(ktitle, fontsize=10)
        d_t = [np.hypot(*m) for m in snaps["transit"]["mates"]]
        d_s = [np.hypot(*m) for m in snaps["search"]["mates"]]
        fig.suptitle(f"위험 구역 채널 (지리산 사례 {snap_case + 1}, R16 두 대): 가운데 삼각형 = 나(위 = 진행 방향), X = 동료 지금 위치 "
                     f"(실선 원 500 m, 점선 원 1 km); 동료 거리 이동 중 {d_t[0]:.0f} m, 탐색 중 {d_s[0]:.0f} m. 밝을수록 1", fontsize=12)
        fig.tight_layout()
        fig.savefig(OUT/"m3_channels.png", dpi=95)
        print(OUT/"m3_channels.png")

    # figure 2: the flights
    fig, ax = plt.subplots(3, 2, figsize=(16, 12))
    for c in range(3):
        a, b = ax[c]
        for dz, col, ls in (("m2", "#8a94a0", "--"), ("m3", "#2b6cb0", "-")):
            d = runs.get((dz, c))
            if d is None:
                continue
            rows = d["rows"]
            t = (np.array([r["t"] for r in rows]) - d["launch_t"][0])/60
            dn = np.array([r["d_now"] for r in rows])/1000
            both = np.array([all(r["active"]) for r in rows])
            a.plot(t[both], dn[both], ls, color=col, lw=1, label=f"{dz} 규칙 (팀 POS {d['found']:.3f})")
            if dz == "m3":
                strict = np.array([any(p[8] or p[9] for p in r["pairs"]) for r in rows])
                if strict.any():
                    a.axvspan(t[np.argmax(strict)], t[both][-1], color="#2b6cb0", alpha=0.07, label="m3 탐색 규칙 (둘 중 하나라도 탐색)")
                cut = np.array([max(r["cut"]) > 0 for r in rows])
                a.plot(t[cut & both], dn[cut & both], "|", color="#c05621", ms=8, label="m3 필터 개입")
                gain = np.cumsum([sum(r["parts"][0]) + sum(r["parts"][1]) for r in rows])
                b.plot(t, gain, color="#2b6cb0", label="탐색 보상 + A′ (두 대 합)")
                for s, sc in (("step", "#c05621"), ("linear", "#d69e2e")):
                    b.plot(t, np.cumsum([_penalties(r, s).sum() for r in rows]), color=sc, label=f"벌점 m3 {s}")
            else:
                b.plot(t, np.cumsum([sum(r["parts"][3]) for r in rows]), "--", color=col, label="벌점 m2 규칙 (그 비행)")
        for y, ls in ((0.5, "-"), (1.0, "--")):
            a.axhline(y, color="k", lw=0.7, ls=ls)
        a.set_ylim(0, 5)
        a.set_ylabel("두 대 사이 거리 (km)")
        a.set_title(f"사례 {c + 1}: 두 대 사이 거리 (실선 0.5 km, 점선 1 km)")
        b.set_title(f"사례 {c + 1}: 누적 탐색 보상과 벌점")
        for x in (a, b):
            x.grid(alpha=0.3)
            x.set_xlabel("첫 이륙 뒤 (분)")
        a.legend(fontsize=8, loc="upper right")
        b.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT/"m3_flights.png", dpi=100)
    print(OUT/"m3_flights.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("fly", "report"))
    ap.add_argument("--case", type=int, default=0)
    ap.add_argument("--design", choices=("m2", "m3"), default="m3")
    ap.add_argument("--model", default=MODEL)
    a = ap.parse_args()
    if a.what == "fly":
        fly(a.case, a.design, a.model)
    else:
        report()


if __name__ == "__main__":
    main()

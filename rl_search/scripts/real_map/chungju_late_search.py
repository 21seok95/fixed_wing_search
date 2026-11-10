"""Why do the last search rectangles come back near the LKP? What is in each rectangle, by the time it is flown.

Flies the agreed settings (lead + gimbal 8 s, 0.7 x model, pass gap 5 s, lane width 190 m, 30 deg bank, lane time flown, transit
limit 30 min, fix A) on scenarios 1-10 and on the web viewer's saved scenario, and records the remaining particle
weights at each plan. For each task, at the middle of its lanes:
  stuck = moved less than ``D`` metres in the last 20 minutes (still on the map); reason from the state in those
  20 minutes: off-road >= half ("논밭"), stopped >= half ("정지"), otherwise on the road network ("도로").

Figure and numbers: rl_search/data/real/chungju/late_search/

    python -m rl_search.scripts.real_map.chungju_late_search --jobs 8
"""
from __future__ import annotations

import argparse
import json
import warnings
from multiprocessing import get_context

import numpy as np

from rl_search.real_terrain import DATA
from rl_search.scripts.real_map.chungju_step3 import in_rect

OUT = DATA/"chungju"/"late_search"
WINDOW_S = 1200.0
DS = (500.0, 2000.0)
WEB = 0


def one(seed, particles=2000):
    warnings.filterwarnings("ignore")
    from rl_search.multi import run_team_launch
    from rl_search.real_scenario import BANK_DEG, W_PLAN, VehicleScenarios, make_env
    from rl_search.sarops import SaropsPlanner
    from rl_search.vehicle import EXIT, OFF, STOP, VehicleConfig, simulate_vehicle
    env = make_env(max(seed, 1), lead_m=510.0, prelook_s=8.0, detection=dict(pass_mode="scale", pass_gap_s=5.0), bank_deg=BANK_DEG)
    env.spacing_budget = True
    if seed == WEB:
        from rl_search.scripts.real_map.vehicle_tuner import load_settings
        cfg, view = load_settings()
        vs = VehicleScenarios(env.sampler, cfg)
        scen = vs.make_at(view["seed"], view["lkp"], view["heading"])
        vc = VehicleConfig(**{**cfg.__dict__, "duration_s": max(cfg.duration_s, env.sampler.cfg.flight_s + 600.0)})
        res = simulate_vehicle(vs.net, vc, np.asarray(view["lkp"], float), float(view["heading"]), np.random.default_rng(view["seed"]))
    else:
        vs = VehicleScenarios(env.sampler)
        scen = vs.make(seed, particles)
        rng = np.random.default_rng(seed)
        r, c = np.divmod(rng.choice(vs.lkp_cells), len(vs.net.terr.axis))
        lkp = np.array([vs.net.terr.axis[c], vs.net.terr.axis[r]], float)
        heading = float(rng.uniform(0, 360))
        vc = VehicleConfig(**{**vs.vcfg.__dict__, "particles": particles, "duration_s": env.sampler.cfg.flight_s + 600.0})
        res = simulate_vehicle(vs.net, vc, lkp, heading, rng)
    assert np.array_equal(res.track.astype(np.float32), scen.track), "state record does not match the scenario"

    class Rec(SaropsPlanner):
        def _plan(self, e):
            w0 = e.belief.w.copy()
            ok = super()._plan(e)
            if ok:
                self.rec.append(w0)
            return ok

    pol = Rec(W=W_PLAN, n_starts=3, big_jump=True, search_s=600.0, max_transit_s=1800.0, lane_mode="cover", peak_timing="intercept")
    pol.rec, pol.team = [], [pol]
    out = run_team_launch([env], [pol], dict(scenario=scen, stage=3), 60.0)
    t_end, dt = float(out["tt"][0][-1]), scen.save_dt_s
    w_win = int(round(WINDOW_S/dt))
    rows, late = [], None
    for k, (task, (tp, r, _)) in enumerate(zip(pol.tasks, pol.rects)):
        if task["lane_t"] is None:
            continue
        te = task["end"] if task["end"] is not None else t_end
        tm = (task["lane_t"] + te)/2
        f = min(int(round(tm/dt)), len(scen.track) - 1)
        a = max(f - w_win, 0)
        P = np.asarray(scen.track[f], float)
        live = res.state[f] != EXIT
        w = pol.rec[k]*live/env.belief.n
        rect = (tp, r.cx, r.cy, r.ell, r.wid, r.theta)
        ins = in_rect(P[:, 0], P[:, 1], rect)
        disp = np.linalg.norm(P - np.asarray(scen.track[a], float), axis=1)
        s = res.state[a:f + 1]
        off, stop = (s == OFF).mean(0), (s == STOP).mean(0)
        reason = np.where(off >= .5, 0, np.where(stop >= .5, 1, 2))
        lkp = np.asarray(scen.lkp, float)
        dl = np.linalg.norm(P - lkp, axis=1)
        m_in, m_all = float(w[ins].sum()), float(w.sum())
        dl10 = np.linalg.norm(np.asarray(scen.track[max(f - int(round(600.0/dt)), 0)], float) - lkp, axis=1)
        tr = np.asarray(scen.track[:f + 1], float)
        path = np.linalg.norm(np.diff(tr, axis=0), axis=2).sum(0)
        near_s = (np.linalg.norm(tr - lkp, axis=2) < 1500.0).sum(0)*dt/60
        kinds = dict(back=dl - dl10 < -500.0, wind=path > 2*dl, slow=near_s >= 15.0)
        row = dict(seed=seed, task=k, tp=tp/60, tm=tm/60, d_lkp=float(np.hypot(r.cx - lkp[0], r.cy - lkp[1])/1000),
                   d_all=float(np.median(dl[w > 0])/1000) if (w > 0).any() else np.nan, m_in=m_in, m_all=m_all, ell=r.ell, wid=r.wid,
                   gain=float(np.interp(te, out["t"], out["pos"]) - np.interp(task["t0"], out["t"], out["pos"])))
        for nm, mask in kinds.items():
            row[f"in_{nm}"], row[f"all_{nm}"] = float(w[ins & mask].sum()/max(m_in, 1e-12)), float(w[mask].sum()/max(m_all, 1e-12))
        row["path_km"] = float(np.average(path[ins]/1000, weights=w[ins])) if m_in > 0 else np.nan
        for D in DS:
            st = disp < D
            row[f"in_{D:.0f}"] = float(w[ins & st].sum()/max(m_in, 1e-12))
            row[f"all_{D:.0f}"] = float(w[st].sum()/max(m_all, 1e-12))
            for i, nm in enumerate(("off", "stop", "road")):
                row[f"in_{D:.0f}_{nm}"] = float(w[ins & st & (reason == i)].sum()/max(m_in, 1e-12))
        rows.append(row)
        late = dict(seed=seed, task=k, rect=rect, P=P, w=w, disp=disp, reason=reason, tm=tm/60, lkp=lkp)
    print(f"seed {seed}: POS {out['pos'][-1]:.3f}, " + ", ".join(f"{x['tp']:.0f}분 {x['d_lkp']:.1f}km 멈춤{x['in_500']:.0%}" for x in rows), flush=True)
    xy = np.asarray(out["xy"][0], float)
    return rows, late, xy, float(out["pos"][-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=[WEB] + list(range(1, 11)), help=f"{WEB} = the web viewer's saved scenario")
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    with get_context("spawn").Pool(args.jobs, maxtasksperchild=1) as pool:
        res = pool.map(one, args.seeds, chunksize=1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    from rl_search.scripts.real_map.chungju_sarops import rect_poly
    plt.rcParams["font.family"] = ["AppleGothic", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False
    rows = [r for x in res for r in x[0]]
    A = {k: np.array([r[k] for r in rows], float) for k in rows[0]}
    late = A["tp"] >= 35
    wm = lambda v, m: float(np.average(v[m], weights=A["m_in"][m]))
    fig = plt.figure(figsize=(21, 12), layout="constrained")
    gs = fig.add_gridspec(2, 3)
    ax = fig.add_subplot(gs[0, 0])
    ax.scatter(A["tp"], A["d_lkp"], c="#2a4d7a", s=34, label="탐색 사각형의 LKP 거리")
    ax.scatter(A["tp"], A["d_all"], c="#bbb", s=22, label="남은 particle의 LKP 거리 (중앙값)")
    ax.set_xlabel("계획 시각 (분)"); ax.set_ylabel("km"); ax.legend(); ax.set_title("사각형이 LKP에서 얼마나 떨어져 있나", fontsize=11)
    ax = fig.add_subplot(gs[0, 1])
    ax.scatter(A["tp"], 100*A["in_500"], c="#c0392b", s=34, label="사각형 안 (500 m)")
    ax.scatter(A["tp"], 100*A["all_500"], c="#bbb", s=22, label="남은 확률 전체 (500 m)")
    ax.set_xlabel("계획 시각 (분)"); ax.set_ylabel("%"); ax.legend(); ax.set_title("20분 동안 500 m 안에 머문 particle의 비중 (남은 확률 기준)", fontsize=11)
    ax = fig.add_subplot(gs[0, 2])
    names = ("off", "stop", "road"); labs = ("논밭에서 맴돎", "정지", "도로 위")
    cols = ("#27ae60", "#8e44ad", "#7f8c8d")
    x = np.arange(4)
    groups = [("전반 500 m", ~late, "500"), ("후반 500 m", late, "500"), ("전반 2 km", ~late, "2000"), ("후반 2 km", late, "2000")]
    bot = np.zeros(4)
    for nm, lab, c in zip(names, labs, cols):
        v = np.array([100*wm(A[f"in_{d}_{nm}"], m) for _, m, d in groups])
        ax.bar(x, v, bottom=bot, color=c, label=lab, width=.6)
        bot += v
    ax.set_xticks(x); ax.set_xticklabels([g[0] for g in groups]); ax.set_ylabel("사각형 안 남은 확률 중 %"); ax.legend()
    ax.set_title("사각형 안의 멈춘 particle: 왜 멈췄나 (후반 = 35분 이후 계획)", fontsize=11)
    order = sorted(range(len(res)), key=lambda i: -res[i][1]["w"][res[i][1]["disp"] < 500].sum()/max(res[i][1]["w"].sum(), 1e-12))[:3]
    for j, i in enumerate(order):
        L = res[i][1]
        ax = fig.add_subplot(gs[1, j])
        P, w = L["P"]/1000, L["w"]
        a = w > 0
        c = np.where(L["disp"] < 500, np.array(cols)[L["reason"]], "#c9d6e8")
        o = np.argsort(L["disp"] < 500)
        o = o[a[o]]
        ax.scatter(P[o, 0], P[o, 1], s=6 + 4000*w[o], c=c[o], lw=0, alpha=.85)
        ax.add_patch(Polygon(rect_poly(*L["rect"][1:]), fill=False, ec="black", lw=2))
        xy = res[i][2]/1000
        ax.plot(xy[:, 0], xy[:, 1], color="#1f4e8c", lw=.8, alpha=.7)
        ax.plot(*(L["lkp"]/1000), "*", ms=15, mfc="yellow", mec="black")
        cx, cy = L["rect"][1]/1000, L["rect"][2]/1000
        span = 10
        ax.set_xlim(cx - span, cx + span); ax.set_ylim(cy - span, cy + span); ax.set_aspect("equal")
        name = "웹 시나리오" if L["seed"] == WEB else f"시나리오 {L['seed']}"
        ax.set_title(f"{name}: 마지막 사각형 ({L['tm']:.0f}분), 연한 파랑 = 움직이는 particle, 나머지 = 멈춤 (색은 위와 같음), 크기 = 남은 확률", fontsize=9)
    fig.suptitle("후반 탐색에 무엇이 남아 있나 (앞섬 + 짐벌 8초, 0.7 × 모델, 통과 간격 5초, 폭 190 m, 이동 제한 30분, 수정 A)", fontsize=13)
    p = OUT/"late_search.png"
    fig.savefig(p, dpi=95)
    summary = dict(n_tasks=len(rows), n_late=int(late.sum()), pos=[x[3] for x in res])
    for lab, m in (("early", ~late), ("late", late)):
        summary[lab] = dict(n=int(m.sum()), d_lkp_km=float(np.mean(A["d_lkp"][m])), d_all_km=float(np.nanmean(A["d_all"][m])), gain=float(np.mean(A["gain"][m])), m_in=float(np.mean(A["m_in"][m])),
                            **{f"{k}": wm(A[k], m) for k in ("in_500", "all_500", "in_2000", "all_2000", "in_500_off", "in_500_stop", "in_500_road", "in_2000_off", "in_2000_stop", "in_2000_road")})
    near = late & (A["d_lkp"] < 10)
    summary["late_near_lkp"] = dict(n=int(near.sum()), **{k: wm(A[k], near) for k in ("in_back", "all_back", "in_wind", "all_wind", "in_slow", "all_slow", "in_500", "in_2000", "path_km")}) if near.any() else {}
    summary["late_far_lkp"] = dict(n=int((late & ~near).sum()), **{k: wm(A[k], late & ~near) for k in ("in_back", "all_back", "in_wind", "all_wind", "in_slow", "all_slow", "in_500", "in_2000", "path_km")})
    for r in rows:
        if r["tp"] >= 35 and r["d_lkp"] < 10:
            print("near", {k: round(r[k], 3) if isinstance(r[k], float) else r[k] for k in ("seed", "task", "tp", "d_lkp", "d_all", "m_in", "gain", "in_back", "in_wind", "in_slow", "in_500", "in_2000", "path_km")})
    (OUT/"late_search.json").write_text(json.dumps(dict(summary=summary, rows=rows), indent=1, ensure_ascii=False))
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    print("Saved", p)


if __name__ == "__main__":
    main()

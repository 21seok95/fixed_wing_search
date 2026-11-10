"""Step 3 check: does the planner's timing match the flight? Per task: planned transit / lane time (what the
accordion assumes: straight line to the nearest corner, ``fixed_lane_time``) against the flown ones, and the
probability in the rectangle over time (weights at the planning moment, particles moving), so a rectangle that is
empty when the UAV is there shows why.

Re-flies the step 3 settings (chungju_step3.run) with a recording planner; the result must equal step3/run_<seed>.npz.
Outputs in rl_search/data/real/chungju/step3/: timing_<seed>.npz, timing_all.png, timing_<seed>.png

    python -m rl_search.scripts.real_map.chungju_timing_check --seeds 1 2 3 4 5 6 7 8 9 10 --jobs 4 --show 9 6
    python -m rl_search.scripts.real_map.chungju_timing_check --lane-time flown       (planner with flown_lane_time)
"""
from __future__ import annotations

import argparse
import warnings
from multiprocessing import get_context

import numpy as np

from rl_search.scripts.real_map.chungju_step3 import LEAD_M, OUT, PRELOOK_S


def rec_run(seed, particles=2000, lane_time="geometry", tag=""):
    warnings.filterwarnings("ignore")
    from rl_search.accordion import fixed_lane_time, flown_lane_time, start_transit, transit_time
    from rl_search.multi import run_team_launch
    from rl_search.real_scenario import BANK_DEG, W_PLAN, VehicleScenarios, make_env
    from rl_search.sarops import SaropsPlanner

    class Rec(SaropsPlanner):
        def _plan(self, env):
            w0 = env.belief.w.copy()
            ok = super()._plan(env)
            if ok:
                r = self.rects[-1][1]
                n = max(1, int(round(r.wid/r.spacing))) if r.spacing else 1
                s = r.wid/n
                r_turn = env.v/(env.turn/env.cfg.decision_dt_s)
                lt = (flown_lane_time(env, r.ell, n, s) if lane_time == "flown" else
                      fixed_lane_time(r.ell, n, s, r_turn, env.v))
                self.rec.append(dict(t=env.t, w=w0, tr_corner=transit_time(env, r), tr_start=start_transit(env, r, 600.0, 600.0),
                                     lane_plan=lt, n=n, s=s))
            return ok

    env = make_env(seed, lead_m=LEAD_M, prelook_s=PRELOOK_S, detection=dict(pass_mode="scale"), bank_deg=BANK_DEG)
    env.lane_time_model = lane_time
    scen = VehicleScenarios(env.sampler).make(seed, particles)
    pol = Rec(W=W_PLAN, n_starts=3, big_jump=True, search_s=600.0, max_transit_s=600.0, lane_mode="cover",
              peak_timing="intercept")
    pol.rec = []
    pol.team = [pol]
    res = run_team_launch([env], [pol], dict(scenario=scen, stage=3), 60.0)
    rects = np.array([(tp, r.cx, r.cy, r.ell, r.wid, r.theta) for tp, r, _ in pol.rects])
    tasks = np.array([[k["t0"], np.nan if k["lane_t"] is None else k["lane_t"], np.nan if k["end"] is None else k["end"],
                       1.0 if k.get("how") == "cap" else 0.0] for k in pol.tasks])
    plan = np.array([[d["t"], d["tr_corner"], d["tr_start"], d["lane_plan"], d["n"], d["s"]] for d in pol.rec])
    W0 = np.array([d["w"] for d in pol.rec], np.float32)
    np.savez_compressed(OUT/f"timing{tag}_{seed}.npz", rects=rects, tasks=tasks, plan=plan, W0=W0, pos_t=res["t"],
                        pos=res["pos"], xy=res["xy"][0], tt=res["tt"][0])
    print(f"seed {seed}: POS {res['pos'][-1]:.3f}", flush=True)
    return seed


def mass_curve(scen, w0, rect, ts):
    from rl_search.mission import ParticleBelief
    from rl_search.scripts.real_map.chungju_step3 import in_rect
    b = ParticleBelief(scen.track, scen.save_dt_s, scen.exit_step, True)
    out = []
    for t in ts:
        P = b.positions(t)
        m = b.active(t) & in_rect(P[:, 0], P[:, 1], rect)
        out.append(float(w0[m].sum()/b.n))
    return np.array(out)


def analyse(seeds, particles, show, tag=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.patches import Polygon
    from rl_search.real_scenario import VehicleScenarios, make_env
    from rl_search.scripts.real_map.chungju_sarops import rect_poly
    from rl_search.scripts.real_map.chungju_step3 import in_rect
    plt.rcParams["font.family"] = ["AppleGothic", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False
    warnings.filterwarnings("ignore")
    vs = VehicleScenarios(make_env(seeds[0]).sampler)
    rows, curves = [], {}
    for s in seeds:
        o = dict(np.load(OUT/f"timing{tag}_{s}.npz"))
        scen = vs.make(s, particles)
        for k in range(len(o["tasks"])):
            t0, t_l, t_e, cap = o["tasks"][k]
            p = o["plan"][k]
            if not np.isfinite(t_l):
                continue
            t_e = t_e if np.isfinite(t_e) else 3600.0
            ts = np.arange(t0, min(t0 + 1800, 3600) + 1, 30.0)
            m = mass_curve(scen, o["W0"][k], o["rects"][k], ts)
            pw = (ts >= t0 + p[1]) & (ts <= t0 + p[1] + p[3])          # planned window (accordion)
            aw = (ts >= t_l) & (ts <= t_e)                              # flown window
            rows.append(dict(seed=s, task=k, tr_plan=p[1], tr_start=p[2], tr_real=t_l - t0, lane_plan=p[3],
                             lane_real=t_e - t_l, capped=cap, poc_plan=m[pw].mean() if pw.any() else np.nan,
                             poc_real=m[aw].mean() if aw.any() else np.nan, poc_now=m[0], last=not np.isfinite(o["tasks"][k][2])))
            curves[(s, k)] = (ts, m, (t0 + p[1], t0 + p[1] + p[3]), (t_l, t_e), scen, o)

    R = {k: np.array([r[k] for r in rows], float) for k in rows[0]}
    full = R["last"] == 0
    fig = plt.figure(figsize=(21, 12), layout="constrained")
    gs = fig.add_gridspec(2, 3)
    ax = fig.add_subplot(gs[0, 0])
    ax.scatter(R["tr_plan"]/60, R["tr_real"]/60, c=np.where(R["task"] == 0, "#c0392b", "#2a4d7a"), s=30)
    lim = [0, max(R["tr_real"].max(), R["tr_plan"].max())/60 + 1]
    ax.plot(lim, lim, "k:", lw=1)
    ax.set_xlabel("계획한 이동 시간 (분, 가까운 모서리까지 직선)"); ax.set_ylabel("실제 이동 시간 (분)")
    e = (R["tr_real"] - R["tr_plan"])/60
    ax.set_title(f"이동 시간: 실제 − 계획 평균 {e.mean():+.1f}분 (빨강 = 첫 작업)", fontsize=11)
    ax = fig.add_subplot(gs[0, 1])
    c = np.where(R["capped"] > 0, "#c0392b", np.where(full, "#2a4d7a", "#bbb"))
    ax.scatter(R["lane_plan"]/60, R["lane_real"]/60, c=c, s=30)
    ax.plot([0, 20], [0, 20], "k:", lw=1); ax.axhline(12, color="#c0392b", lw=.8, ls="--")
    ax.set_xlim(0, 12); ax.set_ylim(0, 20)
    ax.set_xlabel("계획한 레인 시간 (분, " + ("모의 비행" if tag else "fixed_lane_time") + ")"); ax.set_ylabel("실제 레인 시간 (분)")
    d = (R["lane_real"] - R["lane_plan"])[full & (R["capped"] == 0)]/60
    ax.set_title(f"레인 시간 (빨강 = 12분 상한에 잘림, 회색 = 비행 끝에 잘림)\n끝까지 난 작업: 실제 − 계획 평균 {d.mean():+.1f}분, "
                 f"실제/계획 {np.mean((R['lane_real']/R['lane_plan'])[full & (R['capped'] == 0)]):.2f}배", fontsize=10)
    ax = fig.add_subplot(gs[0, 2])
    ax.scatter(R["poc_plan"], R["poc_real"], c=np.where(R["task"] == 0, "#c0392b", "#2a4d7a"), s=30)
    mx = np.nanmax([R["poc_plan"], R["poc_real"]])*1.05
    ax.plot([0, mx], [0, mx], "k:", lw=1)
    ax.set_xlabel("계획한 시간대에 사각형 안 확률 (평균)"); ax.set_ylabel("실제 비행 시간대에 사각형 안 확률 (평균)")
    ratio = np.nanmean(R["poc_real"])/np.nanmean(R["poc_plan"])
    ax.set_title(f"사각형 안 확률: 실제 시간대 / 계획 시간대 = {ratio:.2f} (평균의 비)", fontsize=11)
    ax = fig.add_subplot(gs[1, :])
    for (s, k), (ts, m, pw, aw, _, _) in curves.items():
        if k > 1:
            continue
        ax.plot((ts - ts[0])/60, m/max(m.max(), 1e-9), color="#c0392b" if k == 0 else "#2a4d7a", lw=1, alpha=.5)
    ax.set_xlabel("계획 후 시간 (분)"); ax.set_ylabel("사각형 안 확률 (작업별 최댓값 = 1)")
    ax.set_title("계획 순간의 가중치로 본 사각형 안 확률의 시간 변화 (빨강 = 첫 작업, 파랑 = 둘째 작업): 차량이 지나가며 몇 분 만에 차고 빠진다",
                 fontsize=11)
    fig.suptitle("계획기의 시간 가정 대 실제 비행 (시나리오 %d개, 작업 %d개, 앞섬 + 짐벌 8초, 폭 190 m, 롤 30°, 레인 시간 = %s)" % (len(seeds), len(rows), "모의 비행" if tag else "기하 근사"),
                 fontsize=13)
    fig.savefig(OUT/f"timing{tag}_all.png", dpi=95)
    plt.close(fig)

    for s in show:
        ks = [k for (ss, k) in curves if ss == s][:3]
        fig = plt.figure(figsize=(7*len(ks), 13), layout="constrained")
        gs = fig.add_gridspec(2, len(ks), height_ratios=[1, 1.4])
        for j, k in enumerate(ks):
            ts, m, pw, aw, scen, o = curves[(s, k)]
            ax = fig.add_subplot(gs[0, j])
            ax.plot((ts - ts[0])/60, m, color="black", lw=2)
            ax.axvspan((pw[0] - ts[0])/60, (pw[1] - ts[0])/60, color="#2a4d7a", alpha=.25, label="계획한 레인 시간대")
            ax.axvspan((aw[0] - ts[0])/60, (aw[1] - ts[0])/60, color="#c0392b", alpha=.25, label="실제 레인 시간대")
            ax.set_xlabel("계획 후 시간 (분)"); ax.set_ylabel("사각형 안 확률 (계획 순간 가중치)"); ax.legend(fontsize=9)
            ax.set_title(f"시나리오 {s} 작업 {k + 1} ({ts[0]/60:.0f}분 계획)", fontsize=11)
            ax = fig.add_subplot(gs[1, j])
            rect = o["rects"][k]
            from rl_search.mission import ParticleBelief
            b = ParticleBelief(scen.track, scen.save_dt_s, scen.exit_step, True)
            w0 = o["W0"][k]
            span = max(rect[3], rect[4])/2 + 3000
            win = (rect[1] - span, rect[1] + span, rect[2] - span, rect[2] + span)
            for t, col, lab in ((ts[0], "#888", "계획 순간"), ((pw[0] + pw[1])/2, "#2a4d7a", "계획한 레인 중간"),
                                ((aw[0] + aw[1])/2, "#c0392b", "실제 레인 중간")):
                P = b.positions(t)
                a = b.active(t) & (w0 > 1e-4)
                ax.scatter(P[a, 0]/1000, P[a, 1]/1000, s=3 + 30*w0[a], c=col, alpha=.45, lw=0, label=f"{lab} {t/60:.1f}분")
            ax.add_patch(Polygon(rect_poly(*rect[1:]), fill=False, ec="black", lw=2))
            tt, xy = o["tt"], o["xy"]
            mm = (tt >= ts[0]) & (tt <= aw[1])
            ax.plot(xy[mm, 0]/1000, xy[mm, 1]/1000, color="#27ae60", lw=1.2, label="무인기")
            ax.set_xlim(win[0]/1000, win[1]/1000); ax.set_ylim(win[2]/1000, win[3]/1000); ax.set_aspect("equal")
            ax.legend(fontsize=8, markerscale=3, loc="lower left")
            ax.set_title("particle 위치 (점 크기 = 남은 가중치)", fontsize=11)
        fig.suptitle(f"시나리오 {s}: 계획한 시간대와 실제 시간대의 사각형 안 확률", fontsize=13)
        fig.savefig(OUT/f"timing{tag}_{s}.png", dpi=95)
        plt.close(fig)
    print(f"transit real-plan {e.mean():+.2f} min; lane real/plan {np.mean((R['lane_real']/R['lane_plan'])[full & (R['capped'] == 0)]):.2f} "
          f"(done tasks {int((full & (R['capped'] == 0)).sum())}/{len(rows)}, capped {int((R['capped'] > 0).sum())}); "
          f"poc real/plan {ratio:.2f}")
    for r in rows:
        if r["task"] == 0:
            print(r["seed"], {k: round(float(v), 3) for k, v in r.items() if k not in ("seed", "task")})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=list(range(1, 11)))
    ap.add_argument("--particles", type=int, default=2000)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--show", type=int, nargs="*", default=[9, 6])
    ap.add_argument("--analyse-only", action="store_true")
    ap.add_argument("--lane-time", default="geometry", choices=["geometry", "flown"])
    args = ap.parse_args()
    tag = "" if args.lane_time == "geometry" else "_flown"
    if not args.analyse_only:
        with get_context("spawn").Pool(args.jobs) as pool:
            pool.starmap(rec_run, [(s, args.particles, args.lane_time, tag) for s in args.seeds])
    analyse(args.seeds, args.particles, args.show, tag)


if __name__ == "__main__":
    main()

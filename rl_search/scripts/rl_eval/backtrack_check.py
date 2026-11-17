"""Where the evader particles come back the way they came, and what the new particle rules change (user 2026-11-17: particles should
not come back the way they came; at a dead end they go into the land ahead, and around what they cannot cross). The new target speeds
(spread_check.NEW: paved 50, unpaved 30, off-road 10 km/h, no stops) and the training scenario rules (three types, LKP on a road in the
central 30 km, random heading, route settings drawn from ROUTE) on the 60 km maps, over the first 60 minutes (the targets move at most
60 min once the UAV launches at the report time). The scenarios are spread_check's.

Variants (VARIANTS): now = the vehicle model so far; new = the revised rules (docs/scenario_redesign.md): start towards the goal (with
p_detour the way the heading points), no road driven back and no turning on a dead-end branch, off the road at a dead end (every type),
detours may take a dead-end branch within 45 deg of the escape heading, off-road obstacles gone around; new90 = new, and detours take
only roads within 90 deg of the escape heading.

simulate_vehicle(diag=True) logs the events (rl_search.vehicle._Diag). Per particle, each split by its cause:
  - same road back: it drove a road it had driven, the other way; cause = that of its first such turn.
  - back towards the LKP: it came back towards the LKP by more than 1 km from the farthest it had been (from the tracks); cause = the
    last turn-around that began before then, or none (the road itself bends back, or off-road wandering).
Variety: how far the particles are at 60 min (R50, R95 from the LKP), the length of road any particle drove, and the effective number of
1 km cells they occupy at 60 min (exp of the entropy). Shares are shares of particles (equal weights, so also of the probability).

    python -m rl_search.scripts.rl_eval.backtrack_check --variant now
    python -m rl_search.scripts.rl_eval.backtrack_check --variant new
    python -m rl_search.scripts.rl_eval.backtrack_check --variant new90
    python -m rl_search.scripts.rl_eval.backtrack_check --compare now new new90     # figures from the saved json files
"""
from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import replace
from multiprocessing import get_context
from pathlib import Path

import numpy as np

TYPES = ("road", "mixed", "desert")
TN = {"road": "도로형", "mixed": "혼합형", "desert": "사막형"}
RN = {"amboy": "암보이", "mojave": "바스토"}
MINUTES = 60
BACK_M = 1000.0
NEW = dict(start_by_route=True, no_return=True, deadend_offroad=True, detour_deadend_deg=45.0, off_follow=True)
VARIANTS = {"now": {}, "new": NEW, "new90": NEW | dict(detour_max_deg=90.0)}       # new90 = vehicle.NO_BACKTRACK (agreed 11-17)
VN = {"now": "지금 모델", "new": "수정안", "new90": "수정안 + 우회는 도주 방향 ±90° 안만"}
VC = {"now": "#9a9a95", "new": "#2a78d6", "new90": "#1baf7a"}
CAUSES = (("start", "처음 방향이 목표 반대쪽", "#2a78d6"),
          ("start_spur", "출발점이 막다른 가지 위", "#7a52c7"),
          ("detour", "무작위 우회 뒤", "#eb6834"),
          ("rejoin", "도로 밖에서 돌아온 뒤", "#1baf7a"),
          ("map_edge", "지도 가장자리 500 m 앞에서 끊긴 길", "#d6336c"),
          ("other", "그 밖 (도주 방향 흔들림 등)", "#52514e"),
          ("off_turn", "도로 밖에서 앞이 다 막혀 돌아섬", "#c9772e"),
          ("deadend_off", "막다른 길 끝에서 도로 밖으로 나간 뒤", "#0f8a8a"),
          ("follow", "도로 밖 장애물을 돌아가는 중", "#e0a32e"),
          ("none_road", "되돌아선 일 없음: 도로 자체가 LKP 쪽으로 휨", "#c9c8c1"),
          ("none_off", "되돌아선 일 없음: 도로 밖에서 헤맴", "#e4d6bf"))
CN = {k: v for k, v, _ in CAUSES}
CC = {k: c for k, _, c in CAUSES}
ROAD_CAUSES = ("start", "start_spur", "detour", "rejoin", "map_edge", "other")
TIP_CAUSE = dict(start_on_spur="start_spur", trick_start="start", trick_detour="detour", trick_rejoin="rejoin", rejoin_on_spur="rejoin",
                 trick_other="other", detour_into_spur="other", lost="other", map_edge="map_edge")
HOW_CAUSE = {1: "start", 2: "detour", 3: "rejoin", 0: "other"}
EXAMPLES = (("road", "start"), ("road", "start_spur"), ("road", "detour"), ("road", "rejoin"), ("off", "off_turn"),
            ("geo", "none_road"))
_NET = {}


def _init(regions):
    from rl_search.rl_real import preload
    for r in regions:
        if r not in _NET:
            _NET[r] = preload(r)["net"]


def simulate(region, typ, i, particles, **override):
    from rl_search.rl_eval_set import diverse_lkp, multi_vc, route_vc, vehicle_config
    from rl_search.scripts.rl_eval.spread_check import NEW as SPEEDS
    from rl_search.vehicle import simulate_vehicle
    net = _NET[region]
    s = 7_000_000 + 1000*TYPES.index(typ) + i + (500 if region == "mojave" else 0)
    rng = np.random.default_rng(s)
    vc = multi_vc(rng, typ) | SPEEDS | route_vc(s)
    lkp = diverse_lkp(net, rng)
    heading = float(rng.uniform(0.0, 360.0))
    cfg = replace(replace(vehicle_config(), **vc), particles=particles, duration_s=(MINUTES + 10)*60.0, **override)
    return s, vc, simulate_vehicle(net, cfg, lkp, heading, np.random.default_rng(s), diag=True)


def classify(res):
    """Per particle within the first MINUTES: cause of its first same-road turn (or ""), cause of its coming back towards the LKP
    (or ""), off-road turn-around, dead-end exit, obstacle following and drift flags, the event (step, x, y) that stands for each,
    the distances from the LKP and the time series."""
    from rl_search.vehicle import (EV_DEADEND_OFF, EV_FOLLOW, EV_OFF_TURN, EV_REJOIN_BACK, EV_SAME_ROAD, EV_WANT_BACK, OFF,
                                   TURN_CAUSES)
    cfg = res.cfg
    n = res.track.shape[1]
    per = int(round(cfg.save_dt_s/cfg.dt_s))
    steps = int(round(MINUTES*60/cfg.dt_s))
    ev = res.diag["events"]
    ev = ev[ev[:, 0] < steps]
    ev = ev[np.lexsort((ev[:, 0], ev[:, 4]))]                    # by particle, then time
    kind, p, how = ev[:, 3].astype(int), ev[:, 4].astype(int), ev[:, 5].astype(int)
    cause = np.full(len(ev), "", object)
    tip = kind < len(TURN_CAUSES)
    cause[tip] = [TIP_CAUSE[TURN_CAUSES[k]] for k in kind[tip]]
    for k in (EV_SAME_ROAD, EV_WANT_BACK):
        m = kind == k
        cause[m] = [HOW_CAUSE[h] for h in how[m]]
    cause[kind == EV_REJOIN_BACK] = "rejoin"
    cause[kind == EV_OFF_TURN] = "off_turn"
    cause[kind == EV_DEADEND_OFF] = "deadend_off"
    cause[kind == EV_FOLLOW] = "follow"

    def first_of(mask):
        """Per particle: the first event of this mask (step, x, y), NaN where none."""
        out = np.full((n, 3), np.nan)
        if mask.any():
            _, j = np.unique(p[mask], return_index=True)
            out[p[mask][j]] = ev[mask][j, :3]
        return out

    road_ev = tip | (kind == EV_SAME_ROAD) | (kind == EV_REJOIN_BACK)
    road = np.full(n, "", object)
    road_at = np.full((n, 3), np.nan)
    t_road = np.full(n, np.inf)
    if road_ev.any():
        e, c = ev[road_ev], cause[road_ev]
        _, j = np.unique(e[:, 4].astype(int), return_index=True)
        q = e[j, 4].astype(int)
        road[q], road_at[q], t_road[q] = c[j], e[j, :3], e[j, 0]*cfg.dt_s/60.0
    off_at, dead_at, fol_at = first_of(kind == EV_OFF_TURN), first_of(kind == EV_DEADEND_OFF), first_of(kind == EV_FOLLOW)
    ds = res.diag["drift_step"]
    drift = (ds >= 0) & (ds < steps)
    k_end = int(round(MINUTES*60/cfg.save_dt_s))
    tr = res.track[:k_end + 1].astype(float)
    d = np.hypot(tr[..., 0] - res.lkp_road[0], tr[..., 1] - res.lkp_road[1])
    cm = np.maximum.accumulate(d, axis=0)
    back = cm - d >= BACK_M
    big = (back & (cm - d >= 0.5*cm)).any(axis=0)                  # came back more than half of the farthest it had been
    kb = np.argmax(back, axis=0)
    geo = np.full(n, "", object)
    geo_at = np.full((n, 3), np.nan)
    turn_ev = tip | np.isin(kind, (EV_WANT_BACK, EV_OFF_TURN, EV_REJOIN_BACK, EV_DEADEND_OFF, EV_FOLLOW))
    e, c = ev[turn_ev], cause[turn_ev]
    pe = e[:, 4].astype(int)
    for q in np.flatnonzero(back.any(axis=0)):
        m = (pe == q) & (e[:, 0] <= kb[q]*per)
        if m.any():
            j = np.flatnonzero(m)[-1]
            geo[q], geo_at[q] = c[j], e[j, :3]
        else:
            km = int(np.argmax(d[:kb[q] + 1, q]))                 # where it was farthest before coming back
            geo[q] = "none_off" if res.state[km, q] == OFF else "none_road"
            geo_at[q] = (km*per, tr[km, q, 0], tr[km, q, 1])
    mins = np.arange(MINUTES + 1)
    pm = int(round(60/cfg.save_dt_s))
    cells = np.floor(tr[-1]/1000.0).astype(np.int64)
    _, cnt = np.unique(cells[:, 0]*100003 + cells[:, 1], return_counts=True)
    pr = cnt/cnt.sum()
    out = res.exit_step
    return dict(road=road, road_at=road_at, geo=geo, geo_at=geo_at, off=~np.isnan(off_at[:, 0]), off_at=off_at,
                dead=~np.isnan(dead_at[:, 0]), dead_at=dead_at, fol=~np.isnan(fol_at[:, 0]), fol_at=fol_at, drift=drift, dist=d, big=big,
                ts_geo=np.cumsum(back, axis=0).astype(bool)[::pm].mean(axis=1),
                ts_road=(t_road[None, :] <= mins[:, None]).mean(axis=1),
                exit60=float(((out >= 0) & (out <= k_end)).mean()), cells60=float(np.exp(-(pr*np.log(pr)).sum())),
                r50=float(np.percentile(d[-1], 50)), r95=float(np.percentile(d[-1], 95)), r95_30=float(np.percentile(d[k_end//2], 95)),
                off60=float((res.state[k_end] == OFF).mean()))


def job(a):
    region, typ, i, particles, variant = a
    warnings.filterwarnings("ignore")
    s, vc, res = simulate(region, typ, i, particles, **VARIANTS[variant])
    c = classify(res)
    k_end = int(round(MINUTES*60/res.cfg.save_dt_s))
    e = np.unique(res.edge[:k_end + 1])
    net = _NET[region]
    return dict(region=region, typ=typ, i=i, seed=s, variant=variant,
                road={k: float((c["road"] == k).mean()) for k in ROAD_CAUSES},
                geo={k: float((c["geo"] == k).mean()) for k, _, _ in CAUSES},
                road_n={k: int((c["road"] == k).sum()) for k in ROAD_CAUSES},
                geo_n={k: int((c["geo"] == k).sum()) for k, _, _ in CAUSES},
                off_turn=float(c["off"].mean()), off_n=int(c["off"].sum()), deadend_off=float(c["dead"].mean()),
                follow=float(c["fol"].mean()), drift=float(c["drift"].mean()), big_back=float(c["big"].mean()),
                ts_geo=c["ts_geo"].tolist(), ts_road=c["ts_road"].tolist(), exit60=c["exit60"], particles=len(c["road"]),
                r50=c["r50"], r95=c["r95"], r95_30=c["r95_30"], cells60=c["cells60"], off60=c["off60"],
                road_km=float(net.edge_len[e[e >= 0]].sum()/1000),
                route={k: vc[k] for k in ("heading_std_deg", "reorient_std_deg_min", "p_detour", "goal_sector_deg")},
                leave=vc["p_leave_fast_min"], ret=vc["p_return_min"])


def main():
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--regions", nargs="+", default=["amboy", "mojave"])
    ap.add_argument("--n", type=int, default=12, help="scenarios per map and type (spread_check's)")
    ap.add_argument("--particles", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--variant", default="now", choices=list(VARIANTS))
    ap.add_argument("--figures-only", action="store_true", help="redraw this variant's figures from its saved json")
    ap.add_argument("--compare", nargs="+", help="variants to compare from their saved json files (e.g. now new new90)")
    ap.add_argument("--out", type=Path, default=Path("rl_search/data/real/amboy/rl_eval/report"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.compare:
        runs = {v: json.loads((args.out/f"backtrack_{v}.json").read_text()) for v in args.compare}
        compare(runs, args)
        return
    js = args.out/f"backtrack_{args.variant}.json"
    if args.figures_only:
        rows = json.loads(js.read_text())
    else:
        _init(args.regions)                                      # in the parent: the forked workers share the terrain
        jobs = [(r, t, i, args.particles, args.variant) for r in args.regions for t in TYPES for i in range(args.n)]
        with get_context("fork").Pool(args.workers) as pool:
            rows = pool.map(job, jobs, chunksize=1)
        js.write_text(json.dumps(rows))
    args.n = len([x for x in rows if x["region"] == rows[0]["region"] and x["typ"] == "road"])
    args.particles = rows[0]["particles"]
    summary(rows, args)
    if args.variant == "now":
        examples(rows, args)


def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["AppleGothic", "NanumGothic", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False
    return plt, ("#1f1f1e", "#6b6b67", "#e4e3de", "#fcfcfb")


def _frame(ax, MUTED, GRID, BG, grid_axis="both"):
    ax.grid(axis=grid_axis, color=GRID, lw=.6, zorder=0); ax.set_facecolor(BG); ax.tick_params(colors=MUTED, labelsize=8.5)
    for s_ in ax.spines.values():
        s_.set_color(GRID)


def summary(rows, args):
    plt, (INK, MUTED, GRID, BG) = _style()
    COL = {"amboy": "#2a78d6", "mojave": "#eb6834"}
    regions = list(dict.fromkeys(r["region"] for r in rows))
    groups = [(r, t) for r in regions for t in TYPES]
    sel = lambda r, t: [x for x in rows if x["region"] == r and x["typ"] == t]
    used = [k for k, _, _ in CAUSES if any(x["geo"].get(k, 0) > 0 for x in rows)]
    fig = plt.figure(figsize=(24, 15), facecolor=BG)
    gs = fig.add_gridspec(2, 6, height_ratios=[1.2, 1], hspace=0.3, wspace=0.9, left=0.06, right=0.99, top=0.92, bottom=0.05)
    axA, axB, axC = fig.add_subplot(gs[0, 0:2]), fig.add_subplot(gs[0, 2:4]), fig.add_subplot(gs[0, 4:6])
    y = np.arange(len(groups))
    ylab = [f"{RN[r]} {TN[t]}" for r, t in groups]
    box = dict(facecolor=BG, edgecolor="none", pad=0.6, alpha=0.85)
    for ax, key, keys, title in ((axA, "road", ROAD_CAUSES, "같은 길로 되돌아온 입자 (도로 위)\n색 = 첫 번째로 되돌아온 원인"),
                                 (axB, "geo", used, "LKP 쪽으로 1 km 넘게 되돌아온 입자 (위치로 본 결과)\n색 = 그 전에 시작된 되돌아섬의 원인")):
        left = np.zeros(len(groups))
        for k in keys:
            v = np.array([np.mean([x[key].get(k, 0) for x in sel(r, t)]) for r, t in groups])*100
            ax.barh(y, v, left=left, height=0.62, color=CC[k], label=CN[k], edgecolor=BG, linewidth=1.2, zorder=2)
            left += v
        for g, (r, t) in enumerate(groups):
            tot = np.array([sum(x[key].values()) for x in sel(r, t)])*100
            ax.scatter(tot, np.full(len(tot), g + 0.42), s=10, marker="|", color=INK, alpha=0.55, lw=1.2, zorder=3)
            ax.text(left[g] + 1.0, g, f"{left[g]:.1f}%", va="center", fontsize=9.5, color=INK, bbox=box, zorder=4)
        ax.set_yticks(y, ylab, fontsize=10, color=INK); ax.invert_yaxis()
        ax.set_xlabel(f"입자 몫 (%), 처음 60분. 막대 = 시나리오 {args.n}개 평균, 막대 밑 눈금 = 시나리오 하나하나의 합계", fontsize=9, color=MUTED)
        ax.set_title(title, loc="left", fontsize=11.5, color=INK)
        ax.set_xlim(0, 100)
        ax.legend(fontsize=8.5, frameon=False, loc="lower right", title="원인", title_fontsize=9)
        _frame(ax, MUTED, GRID, BG, "x")
    h = 0.36
    for m, (key, lab, col) in enumerate((("off_turn", "도로 밖에서 앞이 다 막혀 돌아섬", CC["off_turn"]),
                                         ("drift", "도주 방향이 처음보다 90° 넘게 돌아감", "#52514e"))):
        v = np.array([[x[key] for x in sel(r, t)] for r, t in groups])*100
        yy = y + (m - 0.5)*h
        axC.barh(yy, v.mean(1), height=h*0.9, color=col, label=lab, zorder=2)
        for g in range(len(groups)):
            axC.scatter(v[g], np.full(v.shape[1], yy[g]), s=10, marker="|", color=INK, alpha=0.55, lw=1.2, zorder=3)
            axC.text(v[g].mean() + 0.4, yy[g], f"{v[g].mean():.1f}%", va="center", fontsize=9, color=INK, bbox=box, zorder=4)
    axC.set_yticks(y, ylab, fontsize=10, color=INK); axC.invert_yaxis()
    axC.set_xlabel("입자 몫 (%), 처음 60분. 눈금 = 시나리오 하나", fontsize=9, color=MUTED)
    axC.set_title("도로 밖에서 막힘, 도주 방향 흔들림", loc="left", fontsize=11.5, color=INK)
    axC.legend(fontsize=8.5, frameon=False, loc="lower right")
    _frame(axC, MUTED, GRID, BG, "x")
    mins = np.arange(MINUTES + 1)
    for j, t in enumerate(TYPES):
        ax = fig.add_subplot(gs[1, 2*j:2*j + 2])
        for r in regions:
            rr = sel(r, t)
            ax.plot(mins, 100*np.mean([x["ts_road"] for x in rr], 0), color=COL[r], lw=2, label=f"{RN[r]}: 같은 길로 되돌아온 적 있음")
            ax.plot(mins, 100*np.mean([x["ts_geo"] for x in rr], 0), color=COL[r], lw=2, ls="--", label=f"{RN[r]}: LKP 쪽으로 1 km 넘게 되돌아온 적 있음")
        ax.set_title(f"{TN[t]}: 시간에 따라 쌓이는 몫 (시나리오 평균)", loc="left", fontsize=11.5, color=INK)
        ax.set_xlabel("도주 시작 뒤 분", fontsize=9, color=MUTED); ax.set_ylabel("입자 몫 (%)", fontsize=9, color=MUTED)
        ax.set_xlim(0, MINUTES); ax.set_ylim(0, 60)
        ax.legend(fontsize=8.5, frameon=False, loc="upper left")
        _frame(ax, MUTED, GRID, BG)
    fig.suptitle(f"입자가 되돌아오는 경우와 원인 ({VN[args.variant]}, 새 도망자 속도, 60 km 지도, 지도·유형마다 시나리오 {args.n}개 × 입자 {args.particles:,}개)",
                 x=0.01, y=0.985, ha="left", fontsize=15, color=INK)
    fig.savefig(args.out/f"backtrack_{args.variant}.png", dpi=85, facecolor=BG)
    plt.close(fig)
    L = [f"# 입자가 되돌아오는 경우와 원인: {VN[args.variant]} (60 km 지도, 지도·유형마다 시나리오 {args.n}개, 입자 {args.particles:,}개, 처음 {MINUTES}분)", "",
         "값은 입자 몫(%)의 시나리오 평균, 괄호는 가장 큰 시나리오.", "",
         "| 구분 | 원인 | " + " | ".join(ylab) + " |", "|---|---|" + "---|"*len(groups)]
    cell = lambda vals: f"{100*np.mean(vals):.1f} ({100*np.max(vals):.1f})"
    for key, keys, name in (("road", ROAD_CAUSES, "같은 길로 되돌아옴"), ("geo", used, "LKP 쪽으로 1 km 넘게")):
        L.append(f"| {name} | 합계 | " + " | ".join(cell([sum(x[key].values()) for x in sel(r, t)]) for r, t in groups) + " |")
        for k in keys:
            L.append(f"| {name} | {CN[k]} | " + " | ".join(cell([x[key].get(k, 0) for x in sel(r, t)]) for r, t in groups) + " |")
    for key, name in (("off_turn", "도로 밖에서 앞이 다 막혀 돌아섬"), ("deadend_off", "막다른 길 끝에서 도로 밖으로 나감"),
                      ("follow", "도로 밖 장애물을 돌아감"), ("drift", "도주 방향이 90° 넘게 돌아감"), ("exit60", "60분까지 지도 밖으로 나감")):
        L.append(f"| 그 밖 | {name} | " + " | ".join(cell([x[key] for x in sel(r, t)]) for r, t in groups) + " |")
    (args.out/f"backtrack_{args.variant}.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


MEASURES = (("road", "같은 길로 되돌아옴", "%", lambda x: 100*sum(x["road"].values())),
            ("geo", "LKP 쪽으로 1 km 넘게 되돌아옴", "%", lambda x: 100*sum(x["geo"].values())),
            ("big_back", "LKP 쪽으로 크게 되돌아옴 (가장 멀었던 거리의 절반 넘게)", "%", lambda x: 100*x["big_back"]),
            ("off_turn", "도로 밖에서 앞이 다 막혀 돌아섬", "%", lambda x: 100*x["off_turn"]),
            ("deadend_off", "막다른 길 끝에서 도로 밖으로 나감", "%", lambda x: 100*x["deadend_off"]),
            ("r95", "60분에 LKP에서 95%가 드는 거리", "km", lambda x: x["r95"]/1000),
            ("r50", "60분에 LKP에서 50%가 드는 거리", "km", lambda x: x["r50"]/1000),
            ("road_km", "입자가 한 번이라도 달린 도로 길이", "km", lambda x: x["road_km"]),
            ("cells60", "60분에 입자가 퍼진 1 km 칸 수 (실질)", "칸", lambda x: x["cells60"]),
            ("follow", "도로 밖 장애물을 옆으로 돌아감", "%", lambda x: 100*x["follow"]))


def compare(runs, args):
    """Before and after: the measures by map and type for each variant, and four example scenarios."""
    plt, (INK, MUTED, GRID, BG) = _style()
    variants = list(runs)
    rows0 = runs[variants[0]]
    regions = list(dict.fromkeys(r["region"] for r in rows0))
    groups = [(r, t) for r in regions for t in TYPES]
    n_sc = len([x for x in rows0 if x["region"] == regions[0] and x["typ"] == "road"])
    sel = lambda v, r, t: [x for x in runs[v] if x["region"] == r and x["typ"] == t]
    fig, axs = plt.subplots(2, 5, figsize=(30, 12.5), facecolor=BG)
    w = 0.8/len(variants)
    L = [f"# 입자 규칙 전후 비교 (60 km 지도, 지도·유형마다 시나리오 {n_sc}개, 입자 {rows0[0]['particles']:,}개, 처음 {MINUTES}분)", "",
         "값은 시나리오 평균. " + ", ".join(f"{v} = {VN[v]}" for v in variants), "",
         "| 측정 | 모델 | " + " | ".join(f"{RN[r]} {TN[t]}" for r, t in groups) + " |", "|---|---|" + "---|"*len(groups)]
    for ax, (key, title, unit, f) in zip(axs.ravel(), MEASURES):
        x0 = np.arange(len(groups))
        for m, v in enumerate(variants):
            vals = np.array([[f(x) for x in sel(v, r, t)] for r, t in groups])
            xx = x0 + (m - (len(variants) - 1)/2)*w
            ax.bar(xx, vals.mean(1), width=w*0.92, color=VC[v], label=VN[v], zorder=2)
            for g in range(len(groups)):
                ax.scatter(np.full(vals.shape[1], xx[g]), vals[g], s=7, color=INK, alpha=0.35, lw=0, zorder=3)
            L.append(f"| {title} ({unit}) | {VN[v]} | " + " | ".join(f"{vals[g].mean():.1f}" for g in range(len(groups))) + " |")
        ax.set_xticks(x0, [f"{RN[r]}\n{TN[t]}" for r, t in groups], fontsize=8.5, color=INK)
        ax.set_title(f"{title} ({unit})", loc="left", fontsize=11, color=INK)
        _frame(ax, MUTED, GRID, BG, "y")
    axs[0, 0].legend(fontsize=9, frameon=False, loc="upper left")
    fig.suptitle(f"입자 규칙 전후: 되돌아옴과 새 규칙이 하는 일(위 다섯), 퍼짐·다양성(아래 넷)과 장애물 돌아가기. 막대 = 시나리오 {n_sc}개 평균, 점 = 시나리오 하나 (60 km 지도, 처음 60분)",
                 x=0.01, ha="left", fontsize=14, color=INK)
    fig.tight_layout()
    tag = "_".join(variants)
    fig.savefig(args.out/f"backtrack_compare_{tag}.png", dpi=85, facecolor=BG)
    plt.close(fig)
    (args.out/f"backtrack_compare_{tag}.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    compare_examples(runs, args, tag)


def compare_examples(runs, args, tag):
    """Four scenarios where the old model came back most (by cause), flown with each variant: 40 particles' tracks over the hour and
    every particle at 60 min, and the share that came back over time."""
    plt, (INK, MUTED, GRID, BG) = _style()
    from matplotlib.collections import LineCollection
    from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgb
    from matplotlib.ticker import FuncFormatter
    cmap = LinearSegmentedColormap.from_list("t", ["#cfe1f7", "#2a78d6", "#0b2f5e"])
    variants = list(runs)
    rows0 = runs["now"] if "now" in runs else runs[variants[0]]
    picks = []
    for kind, k in (("road", "start"), ("road", "start_spur"), ("road", "detour"), ("off", "off_turn")):
        x = max(rows0, key=lambda r: r["off_n"] if kind == "off" else r["road_n"][k])
        picks.append((k, x))
    nv = len(variants)
    fig = plt.figure(figsize=(6.2*nv + 6.5, 6.4*len(picks)), facecolor=BG)
    gs = fig.add_gridspec(len(picks), nv + 1, width_ratios=[1]*nv + [1.05], hspace=0.3, wspace=0.18, left=0.04, right=0.99,
                          top=0.95, bottom=0.04)
    rng = np.random.default_rng(0)
    for row, (k, x) in enumerate(picks):
        _init([x["region"]])
        net = _NET[x["region"]]
        sims = {v: simulate(x["region"], x["typ"], x["i"], rows0[0]["particles"], **VARIANTS[v])[2] for v in variants}
        cls = {v: classify(res) for v, res in sims.items()}
        k_end = int(round(MINUTES*60/sims[variants[0]].cfg.save_dt_s))
        show = rng.choice(sims[variants[0]].track.shape[1], 40, replace=False)
        P = np.vstack([np.vstack([res.track[:k_end + 1, show].reshape(-1, 2), res.track[k_end]]) for res in sims.values()]).astype(float)
        lo, hi = np.percentile(P, 1, axis=0) - 800, np.percentile(P, 99, axis=0) + 800
        half = max((hi - lo).max()/2, 3000.0)
        cx, cy = (lo + hi)/2
        x0, x1, y0, y1 = cx - half, cx + half, cy - half, cy + half
        t = net.terr
        a = t.axis
        ci = np.flatnonzero((a >= x0 - t.cell) & (a <= x1 + t.cell))
        ri = np.flatnonzero((a >= y0 - t.cell) & (a <= y1 + t.cell))
        step = max(1, len(ci)//500)
        ci, ri = ci[::step], ri[::step]
        img = np.ones((len(ri), len(ci), 3))*np.array(to_rgb(BG))
        for mask, col, al in ((np.asarray(t.slope_deg)[np.ix_(ri, ci)] >= 30, "#c9772e", 0.55), (np.asarray(t.water)[np.ix_(ri, ci)], "#9cc3ee", 0.9),
                              (np.asarray(t.city)[np.ix_(ri, ci)], "#d2d1ca", 0.9)):
            mm = mask.astype(bool)
            img[mm] = (1 - al)*img[mm] + al*np.array(to_rgb(col))
        if not hasattr(net, "_edge_box"):
            po, of = net.poly, net.off
            net._edge_box = np.stack([np.minimum.reduceat(po[:, 0], of[:-1]), np.maximum.reduceat(po[:, 0], of[:-1]),
                                      np.minimum.reduceat(po[:, 1], of[:-1]), np.maximum.reduceat(po[:, 1], of[:-1])], 1)
        b = net._edge_box
        vis = net.edge_main & (b[:, 1] >= x0) & (b[:, 0] <= x1) & (b[:, 3] >= y0) & (b[:, 2] <= y1)
        for col_i, v in enumerate(variants):
            ax = fig.add_subplot(gs[row, col_i])
            res, c = sims[v], cls[v]
            ax.imshow(img, origin="lower", extent=[a[ci[0]] - t.cell/2, a[ci[-1]] + t.cell/2, a[ri[0]] - t.cell/2, a[ri[-1]] + t.cell/2],
                      interpolation="nearest", zorder=0)
            for core, col, lw in ((True, "#c9c8c1", 2.2), (False, "#f0a7b4", 2.6)):
                ks = np.flatnonzero(vis & (net.edge_core == core))
                ax.add_collection(LineCollection([net.poly[net.off[e]:net.off[e + 1]] for e in ks], colors=col, linewidths=lw, zorder=1))
            segs, tt = [], []
            for q in show:
                end = k_end if res.exit_step[q] < 0 else min(k_end, int(res.exit_step[q]))
                tr = res.track[:end + 1, q].astype(float)
                if len(tr) > 1:
                    segs += list(np.stack([tr[:-1], tr[1:]], 1))
                    tt += list(np.arange(len(tr) - 1)*res.cfg.save_dt_s/60)
            lc = LineCollection(segs, cmap=cmap, norm=Normalize(0, MINUTES), linewidths=1.0, alpha=0.9, zorder=3)
            lc.set_array(np.asarray(tt))
            ax.add_collection(lc)
            F = res.track[k_end].astype(float)
            ax.scatter(F[:, 0], F[:, 1], s=1.2, color=INK, alpha=0.25, lw=0, zorder=2)
            L0 = res.lkp_road
            ax.scatter([L0[0]], [L0[1]], marker="*", s=240, color=INK, edgecolors="white", zorder=5)
            same, geo = 100*np.mean(c["road"] != ""), 100*np.mean(c["geo"] != "")
            ax.set_title(f"{VN[v]}\n같은 길로 되돌아옴 {same:.1f}%, LKP 쪽 1 km 넘게 {geo:.1f}%, 60분 R50/R95 {c['r50']/1000:.1f}/{c['r95']/1000:.1f} km",
                         loc="left", fontsize=9.5, color=INK)
            ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect("equal")
            km = FuncFormatter(lambda val, _: f"{val/1000:.0f}")
            ax.xaxis.set_major_formatter(km); ax.yaxis.set_major_formatter(km)
            ax.tick_params(colors=MUTED, labelsize=7.5)
            for s_ in ax.spines.values():
                s_.set_color(GRID)
            if col_i == 0:
                rt = x["route"]
                ax.set_ylabel(f"{RN[x['region']]} {TN[x['typ']]} 시나리오 {x['i'] + 1}\n(지금 모델에서 '{CN[k]}' 가장 많음)\n"
                              f"방향 퍼짐 {rt['heading_std_deg']:.0f}°, 우회 {100*rt['p_detour']:.0f}%/갈림길", fontsize=10, color=INK)
        ax = fig.add_subplot(gs[row, nv])
        mins = np.arange(MINUTES + 1)
        for v in variants:
            ax.plot(mins, 100*cls[v]["ts_geo"], color=VC[v], lw=2, label=f"{VN[v]}: LKP 쪽 1 km 넘게")
            ax.plot(mins, 100*cls[v]["ts_road"], color=VC[v], lw=1.6, ls="--", label=f"{VN[v]}: 같은 길로")
        ax.set_xlim(0, MINUTES); ax.set_ylim(0, 100)
        ax.set_xlabel("도주 시작 뒤 분", fontsize=9, color=MUTED); ax.set_ylabel("되돌아온 적 있는 입자 (%)", fontsize=9, color=MUTED)
        ax.set_title("시간에 따라 쌓이는 되돌아옴", loc="left", fontsize=10, color=INK)
        if row == 0:
            ax.legend(fontsize=8, frameon=False, loc="upper left")
        _frame(ax, MUTED, GRID, BG)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(0, MINUTES))
    cax = fig.add_axes([0.3, 0.012, 0.35, 0.006])
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_label("입자 길 40개의 색 = 도주 시작 뒤 분 (점 = 60분에 모든 입자, ★ = LKP, 회색 선 = 도로, 분홍 = 막다른 가지, 갈색 = 경사 30° 이상)",
                 fontsize=10, color=INK)
    fig.suptitle("같은 시나리오를 세 모델로: 지금 모델에서 되돌아옴이 가장 많던 시나리오 넷 (축은 km)", x=0.01, ha="left", fontsize=14, color=INK)
    fig.savefig(args.out/f"backtrack_compare_{tag}_examples.png", dpi=72, facecolor=BG)
    plt.close(fig)


def examples(rows, args):
    """One example per case: the scenario where the case is most common, up to 8 such particles near each other. Map with the
    tracks, and under it their distance from the LKP over the whole hour."""
    plt, (INK, MUTED, GRID, BG) = _style()
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("t", ["#cfe1f7", "#2a78d6", "#0b2f5e"])
    cnt = lambda x, kind, k: x["off_n"] if kind == "off" else (x["road_n"][k] if kind == "road" else x["geo_n"][k])
    fig = plt.figure(figsize=(22, 20), facecolor=BG)
    gs = fig.add_gridspec(4, 3, height_ratios=[3, 1, 3, 1], hspace=0.42, wspace=0.22, left=0.05, right=0.99, top=0.93, bottom=0.07)
    for n_, (kind, k) in enumerate(EXAMPLES):
        r0, c0 = 2*(n_//3), n_ % 3
        ax, axd = fig.add_subplot(gs[r0, c0]), fig.add_subplot(gs[r0 + 1, c0])
        x = max(rows, key=lambda r: cnt(r, kind, k))
        if cnt(x, kind, k) == 0:
            ax.axis("off"); axd.axis("off")
            continue
        _init([x["region"]])
        s, vc, res = simulate(x["region"], x["typ"], x["i"], args.particles, **VARIANTS[args.variant])
        c = classify(res)
        ids, at = ((np.flatnonzero(c["off"]), c["off_at"]) if kind == "off" else
                   (np.flatnonzero(c["road"] == k), c["road_at"]) if kind == "road" else (np.flatnonzero(c["geo"] == k), c["geo_at"]))
        drawn, t0, t1 = _example(ax, _NET[x["region"]], res, ids, at, k, cmap, (INK, MUTED, GRID, BG))
        mins = np.arange(c["dist"].shape[0])*res.cfg.save_dt_s/60
        for q in drawn:
            axd.plot(mins, c["dist"][:, q]/1000, color=INK, lw=1, alpha=0.55)
            axd.plot(at[q, 0]*res.cfg.dt_s/60, np.interp(at[q, 0]*res.cfg.dt_s/60, mins, c["dist"][:, q]/1000),
                     marker="o" if k == "off_turn" or k.startswith("none") else "x", color="#eb6834" if k == "off_turn" else INK,
                     ms=6, mew=1.6, mfc="none" if k.startswith("none") else None)
        axd.axvspan(t0, t1, color="#cfe1f7", alpha=0.5, lw=0, zorder=0)
        axd.set_xlim(0, MINUTES); axd.set_ylim(bottom=0)
        axd.set_xlabel("도주 시작 뒤 분 (파란 띠 = 위 지도에 그린 시간)", fontsize=8.5, color=MUTED)
        axd.set_ylabel("LKP에서 거리 (km)", fontsize=8.5, color=MUTED)
        _frame(axd, MUTED, GRID, BG)
        rt = x["route"]
        lab = (f"같은 길로 되돌아옴: {CN[k]}" if kind == "road" else CN[k] if kind == "off" else
               "LKP 쪽으로 되돌아옴: " + CN[k].replace("되돌아선 일 없음: ", "되돌아선 일 없이 "))
        ax.set_title(f"{n_ + 1}. {lab}\n{RN[x['region']]} {TN[x['typ']]} 시나리오 {x['i'] + 1}: {cnt(x, kind, k):,}/{x['particles']:,}개 "
                     f"({100*cnt(x, kind, k)/x['particles']:.1f}%), 그린 입자 {len(drawn)}개, {t0:.0f}~{t1:.0f}분\n"
                     f"경로 설정: 방향 퍼짐 {rt['heading_std_deg']:.0f}°, 갈림길 우회 {100*rt['p_detour']:.0f}%, 목표 구역 폭 {rt['goal_sector_deg']:.0f}°",
                     loc="left", fontsize=9.5, color=INK)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    cax = fig.add_axes([0.3, 0.025, 0.4, 0.008])
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal", ticks=[0, 1])
    cb.ax.set_xticklabels(["그린 시간의 처음", "그린 시간의 끝"], fontsize=9)
    cb.set_label("입자 길의 색: 연한 색 → 진한 색 순서로 움직임, 화살촉 = 마지막 진행 방향", fontsize=10, color=INK)
    fig.suptitle("경우마다 가장 많이 일어난 시나리오의 예\n도로: 회색 = 빠져나갈 길이 있는 도로, 분홍 = 막다른 가지 · 땅: 갈색 = 경사 30° 이상 (도로 밖 통행 불가), 파랑 = 물 · "
                 "× = 같은 길로 돌아서기 시작한 곳, ● = 도로 밖에서 막힌 곳, ○ = LKP에서 가장 멀었던 곳, ★ = LKP (지도 밖이면 가장자리 화살표)",
                 x=0.01, y=0.985, ha="left", fontsize=12.5, color=INK)
    fig.savefig(args.out/f"backtrack_{args.variant}_examples.png", dpi=80, facecolor=BG)
    plt.close(fig)


def _example(ax, net, res, ids, at, k, cmap, pal):
    """Tracks of up to 8 particles of one case near each other, on the roads and the off-road obstacles. Window: from the start for
    the start causes, otherwise from 6 min before the event; until 6 min after it. Returns the drawn particles and the window."""
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize, to_rgb
    from matplotlib.ticker import FuncFormatter
    INK, MUTED, GRID, BG = pal
    cfg = res.cfg
    per = cfg.save_dt_s/cfg.dt_s
    xy = at[ids, 1:3]
    c0 = np.median(xy, axis=0)
    ids = ids[np.argsort(np.hypot(*(xy - c0).T))][:8]
    segs, tt, pts, t_lo, t_hi, drawn, heads = [], [], [], [], [], [], []
    for q in ids:
        ke = int(round(at[q, 0]/per))
        k0 = 0 if k in ("start", "start_spur") else max(0, ke - int(6*60/cfg.save_dt_s))
        k1 = min(res.track.shape[0] - 1, ke + int(6*60/cfg.save_dt_s), int(MINUTES*60/cfg.save_dt_s))
        if res.exit_step[q] >= 0:
            k1 = min(k1, int(res.exit_step[q]))
        tr = res.track[k0:k1 + 1, q].astype(float)
        if len(tr) < 2:
            continue
        f = np.linspace(0, 1, len(tr))
        segs += list(np.stack([tr[:-1], tr[1:]], 1))
        tt += list((f[:-1] + f[1:])/2)
        pts.append(tr)
        drawn.append(q)
        heads.append(tr)
        t_lo.append(k0*cfg.save_dt_s/60); t_hi.append(k1*cfg.save_dt_s/60)
    P = np.vstack(pts + [at[drawn, 1:3]])
    lo, hi = np.percentile(P, 0.5, axis=0) - 400, np.percentile(P, 99.5, axis=0) + 400
    half = max((hi - lo).max()/2, 1500.0)
    cx, cy = (lo + hi)/2
    x0, x1, y0, y1 = cx - half, cx + half, cy - half, cy + half
    t = net.terr
    a = t.axis
    ci = np.flatnonzero((a >= x0 - t.cell) & (a <= x1 + t.cell))
    ri = np.flatnonzero((a >= y0 - t.cell) & (a <= y1 + t.cell))
    step = max(1, len(ci)//500)
    ci, ri = ci[::step], ri[::step]
    sub = lambda m: np.asarray(m)[np.ix_(ri, ci)]
    img = np.ones((len(ri), len(ci), 3))*np.array(to_rgb(BG))
    for mask, col, al in ((sub(t.slope_deg) >= cfg.off_max_slope_deg, "#c9772e", 0.6), (sub(t.water), "#9cc3ee", 0.9),
                          (sub(t.city), "#d2d1ca", 0.9), (sub(t.forest), "#1baf7a", 0.35)):
        m = mask.astype(bool)
        img[m] = (1 - al)*img[m] + al*np.array(to_rgb(col))
    ax.imshow(img, origin="lower", extent=[a[ci[0]] - t.cell/2, a[ci[-1]] + t.cell/2, a[ri[0]] - t.cell/2, a[ri[-1]] + t.cell/2],
              interpolation="nearest", zorder=0)
    if not hasattr(net, "_edge_box"):
        po, of = net.poly, net.off
        net._edge_box = np.stack([np.minimum.reduceat(po[:, 0], of[:-1]), np.maximum.reduceat(po[:, 0], of[:-1]),
                                  np.minimum.reduceat(po[:, 1], of[:-1]), np.maximum.reduceat(po[:, 1], of[:-1])], 1)
    b = net._edge_box
    vis = net.edge_main & (b[:, 1] >= x0) & (b[:, 0] <= x1) & (b[:, 3] >= y0) & (b[:, 2] <= y1)
    for core, col, lw in ((True, "#bdbcb6", 3.2), (False, "#f0a7b4", 4.0)):        # wide and light, so the tracks on them show
        ks = np.flatnonzero(vis & (net.edge_core == core))
        ax.add_collection(LineCollection([net.poly[net.off[e]:net.off[e + 1]] for e in ks], colors=col, linewidths=lw, zorder=1,
                                         capstyle="round"))
    lc = LineCollection(segs, cmap=cmap, norm=Normalize(0, 1), linewidths=1.3, alpha=0.95, zorder=3)
    lc.set_array(np.asarray(tt))
    ax.add_collection(lc)
    for tr in heads:                                            # arrowhead: the last direction of travel
        j = len(tr) - 1
        i0 = max(0, j - 3)
        if np.hypot(*(tr[j] - tr[i0])) > 1:
            ax.annotate("", xy=tr[j], xytext=tr[i0], arrowprops=dict(arrowstyle="-|>", color="#0b2f5e", lw=1.2, mutation_scale=14), zorder=4)
    E = at[drawn, 1:3]
    if k == "off_turn":
        ax.scatter(E[:, 0], E[:, 1], marker="o", s=46, color="#eb6834", edgecolors="white", lw=1.2, zorder=5)
    elif k.startswith("none"):
        ax.scatter(E[:, 0], E[:, 1], marker="o", s=70, facecolors="none", edgecolors=INK, lw=1.6, zorder=5)
    else:
        ax.scatter(E[:, 0], E[:, 1], marker="x", s=55, color=INK, lw=1.9, zorder=5)
    L = res.lkp_road
    if x0 <= L[0] <= x1 and y0 <= L[1] <= y1:
        ax.scatter([L[0]], [L[1]], marker="*", s=260, color=INK, edgecolors="white", zorder=6)
    else:                                                       # arrow at the window edge towards the LKP
        v = L - np.array([cx, cy])
        u = v/np.hypot(*v)
        s_ = 0.9*half/max(abs(u[0]), abs(u[1]))
        tip = np.array([cx, cy]) + u*s_
        ax.annotate(f"LKP 쪽 {np.hypot(*v)/1000:.0f} km", xy=tip, xytext=tip - u*0.22*half, fontsize=9, color=INK, ha="center",
                    va="center", arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.4), zorder=6,
                    bbox=dict(facecolor=BG, edgecolor="none", pad=1, alpha=0.85))
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect("equal")
    km = FuncFormatter(lambda v, _: f"{v/1000:.1f}")
    ax.xaxis.set_major_formatter(km); ax.yaxis.set_major_formatter(km)
    ax.set_xlabel("동쪽 (km)", fontsize=8.5, color=MUTED); ax.set_ylabel("북쪽 (km)", fontsize=8.5, color=MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    for s_ in ax.spines.values():
        s_.set_color(GRID)
    return drawn, (min(t_lo) if t_lo else 0), (max(t_hi) if t_hi else 0)


if __name__ == "__main__":
    main()

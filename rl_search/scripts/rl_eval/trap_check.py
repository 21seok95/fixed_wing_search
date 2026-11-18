"""Do off-road targets get trapped in fields on a Korean map? (user 2026-11-18: can the mixed type be trained on Korean maps, where
particles used to get stuck in paddy fields; check the agreed particle rules vehicle.NO_BACKTRACK against the earlier ones.)

Chungju 60 km against the two desert maps (amboy, mojave 60 km), mixed and desert types, the redesigned speeds and route settings
(rl_eval_set.v2_vc without the rules), 20 scenarios per map and type, the same seeds with the earlier rules (vehicle.EARLIER_RULES) and the
agreed ones (NO_BACKTRACK), 2,000 particles, 60 min. Off the road a vehicle may drive where vehicle.RoadNet.set_drive_mask allows (not
forest, water, a city, slope 15 deg or more).

Per particle: "stuck" = off the road for 10 min or more within 300 m of where those 10 min began (no progress); "stuck long" = 30 min.
Also the share of the particle time off the road, and the size of the drivable piece (connected drivable cells) the stuck particles are in.

    python -m rl_search.scripts.rl_eval.trap_check run --n 20 --workers 8
    python -m rl_search.scripts.rl_eval.trap_check figure
"""
from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import replace
from multiprocessing import get_context
from pathlib import Path

import numpy as np

OUT = Path("rl_search/data/real/amboy/rl_eval/report")
WORK = OUT/"trap_check"
REGIONS = ("chungju", "amboy", "mojave")
RN = {"chungju": "충주", "amboy": "암보이", "mojave": "바스토"}
TYPES = ("mixed", "desert")
TN = {"mixed": "혼합형", "desert": "사막형"}
RULES = ("earlier", "agreed", "shortcut")
RL = {"earlier": "예전 규칙", "agreed": "합의한 새 규칙", "shortcut": "새 규칙 + 지름길"}
SHORTCUT = dict(shortcut_m=2000.0, shortcut_factor=1.5)      # user 2026-11-18: off the road only as a shortcut to another road
_NET = {}


def _init():
    from rl_search.rl_real import preload
    for r in REGIONS:
        _NET[r] = preload(r)["net"]


def stuck_stats(track, state, dt, win_min=10.0, radius=300.0):
    from rl_search.vehicle import OFF
    T, N = state.shape
    w = int(round(win_min*60/dt))
    off = state == OFF
    stuck = np.zeros(N, bool)
    stuck_long = np.zeros(N, bool)
    stuck_time = np.zeros(N)
    where = []
    for j in np.flatnonzero(off.any(axis=0)):
        o = off[:, j]
        run_ = 0
        for k in range(T):
            run_ = run_ + 1 if o[k] else 0
            if run_ > w and np.hypot(*(track[k, j] - track[k - w, j])) < radius:
                if not stuck[j]:
                    where.append(track[k, j].copy())
                stuck[j] = True
                stuck_time[j] += dt
        stuck_long[j] = stuck_time[j] >= 20*60
    return dict(off_share=float(off.mean()), stuck=float(stuck.mean()), stuck_long=float(stuck_long.mean()),
                stuck_time_share=float(stuck_time.sum()/max(off.sum()*dt, 1.0)), where=np.array(where).reshape(-1, 2),
                ids=np.flatnonzero(stuck_long if stuck_long.any() else stuck))


def job(a):
    warnings.filterwarnings("ignore")
    region, typ, rules, i = a
    from rl_search.rl_eval_set import diverse_lkp, multi_vc, route_vc, vehicle_config, V2_SPEEDS
    from rl_search.vehicle import EARLIER_RULES, NO_BACKTRACK, simulate_vehicle
    net = _NET[region]
    s = 8_000_000 + 1000*TYPES.index(typ) + i + 300*REGIONS.index(region)
    rng = np.random.default_rng(s)
    vc = multi_vc(rng, typ) | V2_SPEEDS | route_vc(s) | (EARLIER_RULES if rules == "earlier" else NO_BACKTRACK) \
        | (SHORTCUT if rules == "shortcut" else {})
    lkp = diverse_lkp(net, rng, 10000.0)
    heading = float(rng.uniform(0.0, 360.0))
    cfg = replace(replace(vehicle_config(), **vc), particles=2000, duration_s=3600.0, exit_straight=True)
    res = simulate_vehicle(net, cfg, lkp, heading, np.random.default_rng(s))
    st = stuck_stats(res.track.astype(float), res.state, float(cfg.save_dt_s))
    k = min(360, res.track.shape[0] - 1)
    from rl_search.scripts.rl_eval.branch_check import branches
    br = {m: branches(res.track[min(int(m*60/cfg.save_dt_s), res.track.shape[0] - 1)].astype(float)) for m in (30, 60)}
    out = dict(region=region, typ=typ, rules=rules, i=i, n_shortcut=int(len(getattr(res, "shortcut_len", []))),
               shortcut_len=[float(x) for x in getattr(res, "shortcut_len", [])][:400], branches30=br[30][0], branches60=br[60][0], eff60=br[60][1], **{k_: v for k_, v in st.items() if k_ not in ("where", "ids")}, where=st["where"].tolist())
    ids = st["ids"] if rules != "shortcut" else np.flatnonzero((res.state == 2).any(axis=0))   # shortcuts: particles that took one
    if i < 4 and len(ids):                                  # examples: tracks of up to 12 stuck particles (or shortcut takers)
        sel = ids[:12]
        out["ex_track"] = res.track[::3, sel].astype(np.float32).tolist()
        out["ex_state"] = res.state[::3, sel].tolist()
        out["lkp"] = [float(v) for v in res.lkp_road]
    return out


def run(n, workers, rules=RULES, regions=REGIONS):
    WORK.mkdir(parents=True, exist_ok=True)
    _init()
    jobs = [(r, t, ru, i) for r in regions for t in TYPES for ru in rules for i in range(n)]
    with get_context("fork").Pool(workers) as pool:
        rows = []
        for j, x in enumerate(pool.imap_unordered(job, jobs, chunksize=1), 1):
            rows.append(x)
            if j % 30 == 0:
                print(f"{j}/{len(jobs)}", flush=True)
    f = WORK/"rows.json"
    old = [x for x in json.loads(f.read_text()) if (x["rules"], x["region"]) not in {(ru, r) for ru in rules for r in regions}] if f.exists() else []
    f.write_text(json.dumps(old + rows))


def figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import ndimage
    from rl_search.rl_real import preload
    plt.rcParams["font.family"] = ["AppleGothic", "NanumGothic", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False
    INK, MUTED, GRID, BG = "#1f1f1e", "#6b6b67", "#e4e3de", "#fcfcfb"
    rows = json.loads((WORK/"rows.json").read_text())
    fig = plt.figure(figsize=(24, 15), facecolor=BG)
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.2])
    for col, (key, ttl) in enumerate((("stuck", "10분 넘게 300 m 안에 갇힌 적 있는 입자 (%)"), ("stuck_long", "갇힌 시간이 20분 넘는 입자 (%)"),
                                      ("off_share", "입자 시간 중 도로 밖 몫 (%)"))):
        ax = fig.add_subplot(gs[0, col])
        pos, ticks = 0, []
        for r in REGIONS:
            for t in TYPES:
                for k_, ru in enumerate(RULES):
                    v = [x[key] for x in rows if x["region"] == r and x["typ"] == t and x["rules"] == ru]
                    if not v:
                        continue
                    ax.bar(pos + 0.3*k_, 100*np.mean(v), width=0.28, color=("#9a9a95", "#2a78d6", "#1baf7a")[k_],
                           label=RL[ru] if (r, t) == (REGIONS[0], TYPES[0]) else None)
                    ax.scatter(np.full(len(v), pos + 0.3*k_), 100*np.array(v), s=6, color=INK, alpha=0.4, zorder=3)
                ticks.append((pos + 0.3, f"{RN[r]}\n{TN[t]}"))
                pos += 1.2
            pos += 0.4
        ax.set_xticks([p for p, _ in ticks], [l for _, l in ticks], fontsize=8.5)
        ax.set_title(ttl, loc="left", fontsize=10.5, color=INK)
        if col == 0:
            ax.legend(fontsize=9, frameon=False)
    ax = fig.add_subplot(gs[0, 3])
    for t_, col_ in zip(TYPES, ("#2a78d6", "#eb6834")):
        L = np.concatenate([x.get("shortcut_len", []) for x in rows if x["region"] == "chungju" and x["typ"] == t_ and x["rules"] == "shortcut"] or [[]])
        if len(L):
            ax.hist(L, bins=np.arange(0, 2100, 100), color=col_, alpha=0.6, label=f"충주 {TN[t_]} (중앙값 {np.median(L):.0f} m)")
    ax.set_xlabel("지름길 길이 (m)", fontsize=9, color=MUTED); ax.set_title("지름길 길이 (최대 2 km)", loc="left", fontsize=10.5, color=INK)
    ax.legend(fontsize=8.5, frameon=False)
    # examples on Chungju: drivable land, roads, example tracks (off-road parts in colour), stuck points
    net = preload("chungju")["net"]
    net.set_drive_mask(15.0)
    t = net.terr
    gsb = gs[1, :].subgridspec(1, 3)
    for col, ru in enumerate(RULES):
        ax = fig.add_subplot(gsb[0, col])
        ex = [x for x in rows if x["region"] == "chungju" and x["typ"] == "mixed" and x["rules"] == ru and "ex_track" in x]
        e = max(ex, key=lambda x: len(x["ex_track"][0]) if x["ex_track"] else 0)
        tr = np.array(e["ex_track"]); stt = np.array(e["ex_state"])
        if ru == "shortcut":                                # zoom on the longest shortcut of the examples
            offp = [(np.flatnonzero(stt[:, j] == 2), j) for j in range(tr.shape[1])]
            ks, jj = max(offp, key=lambda x: len(x[0]))
            c = tr[ks, jj].mean(axis=0) if len(ks) else tr[-1, 0]
            h = 900.0
        else:
            ends = tr[-1]                                   # zoom on where most of the stuck particles end (densest 2 km)
            cnt = [(np.hypot(*(ends - q).T) < 2000).sum() for q in ends]
            c = ends[int(np.argmax(cnt))]
            h = 2500.0
        ia = lambda v: int(np.clip(np.round((v - t.axis[0])/t.cell), 0, len(t.axis) - 1))
        r0, r1, c0, c1 = ia(c[1] - h), ia(c[1] + h), ia(c[0] - h), ia(c[0] + h)
        dm = net.drive[r0:r1, c0:c1]
        ext = [t.axis[c0]/1000, t.axis[c1]/1000, t.axis[r0]/1000, t.axis[r1]/1000]
        base = np.where(dm, 0.97, np.where(t.forest[r0:r1, c0:c1], 0.45, 0.7))
        ax.imshow(base, origin="lower", extent=ext, cmap="Greys_r", vmin=0, vmax=1)
        rd = t.surface[r0:r1, c0:c1] > 0
        ax.imshow(np.ma.masked_where(~rd, rd.astype(float)), origin="lower", extent=ext, cmap="autumn", alpha=0.9)
        for j in range(tr.shape[1]):
            off = stt[:, j] == 2
            ax.plot(tr[:, j, 0]/1000, tr[:, j, 1]/1000, color="#2a78d6", lw=0.6, alpha=0.6)
            ax.plot(np.where(off, tr[:, j, 0], np.nan)/1000, np.where(off, tr[:, j, 1], np.nan)/1000, color="#1baf7a", lw=1.6)
            ax.plot(*(tr[-1, j]/1000), marker="o", ms=4, color="#d6336c")
        ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3]); ax.set_aspect("equal")
        ax.set_title(f"충주 혼합형, {RL[ru]}: {'지름길을 탄' if ru == 'shortcut' else '갇힌'} 입자 {tr.shape[1]}개의 경로 (확대)\n흰색 = 도로 밖으로 갈 수 있는 땅(논밭 등), 짙은 회색 = 숲, 옅은 회색 = 가파른 곳·물, "
                     f"빨강·노랑 = 도로; 파랑 = 도로 위, 초록 = 도로 밖, 분홍 = 60분 위치", loc="left", fontsize=10, color=INK)
    for a_ in fig.axes:
        a_.set_facecolor(BG); a_.tick_params(colors=MUTED, labelsize=8)
        for s in a_.spines.values():
            s.set_color(GRID)
    fig.suptitle("한국 지형(충주)에서 도로 밖 표적이 갇히는가: 예전 규칙 대 합의한 새 규칙 (지도·유형마다 20개, 입자 2,000개, 60분)", x=0.01, ha="left", fontsize=13.5,
                 color=INK)
    fig.tight_layout()
    fig.savefig(OUT/"trap_check.png", dpi=80, facecolor=BG)
    L = ["# 도로 밖 갇힘", "", "| 지도 | 유형 | 규칙 | 갇힌 적 있음 | 20분 넘게 갇힘 | 도로 밖 시간 중 갇힘 | 도로 밖 몫 |", "|---|---|---|---|---|---|---|"]
    for r in REGIONS:
        for t_ in TYPES:
            for ru in RULES:
                v = [x for x in rows if x["region"] == r and x["typ"] == t_ and x["rules"] == ru]
                L.append(f"| {RN[r]} | {TN[t_]} | {RL[ru]} | {100*np.mean([x['stuck'] for x in v]):.1f}% | {100*np.mean([x['stuck_long'] for x in v]):.1f}% | "
                         f"{100*np.mean([x['stuck_time_share'] for x in v]):.1f}% | {100*np.mean([x['off_share'] for x in v]):.1f}% |")
    (OUT/"trap_check.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=("run", "figure"))
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--rules", nargs="+", default=list(RULES))
    ap.add_argument("--regions", nargs="+", default=list(REGIONS))
    a = ap.parse_args()
    run(a.n, a.workers, a.rules, a.regions) if a.what == "run" else figure()


if __name__ == "__main__":
    main()

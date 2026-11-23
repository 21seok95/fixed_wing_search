"""Lead candidates without particle identities, and particles that look stopped (user 2026-11-19, follow-up of lead_gap.py).

1. Stopped-looking particles: per map and lead, the share of the probability whose move over the lead is under one map cell (a moving
   target that the map shows in the same cell now and after the lead; the targets never stop since R10).
2. Identity-free estimate: the map only shows how many particles are in each cell. Per scene and map, the particles read on that map
   (the smallest map holding them now) are scored; the particles inside the map now (up to 400) are snapped to the map cells now and
   after each lead, the labels dropped, and the two equal sets matched by the least total squared move (scipy linear_sum_assignment);
   C3 chains the second lead to the first. The meeting point then follows as in lead_gap.py. Compared with the same
   particles' estimate with identities. C2 (the meeting point map) needs no matching: only its cell snapping.

    python -m rl_search.scripts.rl_eval.lead_gap2 --pool 40
Writes report/lead_gap2.png and .json.
"""
from __future__ import annotations

import argparse
import itertools
import json

import numpy as np
from scipy.optimize import linear_sum_assignment

from rl_search.scripts.rl_eval.lead_gap import DT, ERR_M, OUT, SIDES, at, flight_scenes, meet, pool_scenes

CANDS = {"C0 지금": [[15, 30, 60, 150, 300, 600, 600]],
         "C3 두 장 (지금, 2배)": [[15, 30, 60, 150, 300, 600, 600], [30, 60, 120, 300, 600, 1200, 1200]]}
N_MATCH = 400


def snap(p, x, c, s, side):
    """Map-cell centres of world points p on the heading-up square map of this side around x."""
    cell = side/64
    dx, dy = p[:, 0] - x[0], p[:, 1] - x[1]
    f, r = dx*c + dy*s, dx*s - dy*c                     # forward, right
    f, r = (np.floor(f/cell) + .5)*cell, (np.floor(r/cell) + .5)*cell
    return np.stack([x[0] + f*c + r*s, x[1] + f*s - r*c], 1)


def scene(P, f0, x, hd, w, act, rng):
    keep = act & (w > 0)
    if keep.sum() < 50 or len(P) - f0 < 3:
        return None
    Pk, wk = P[:, keep], w[keep]
    n_tau = len(P) - f0
    k_true, q, ok = meet(lambda k: Pk[f0 + k], x, hd, n_tau)
    c, s = np.cos(hd), np.sin(hd)

    def cheb(p):
        dx, dy = p[:, 0] - x[0], p[:, 1] - x[1]
        return np.maximum(np.abs(dx*c + dy*s), np.abs(-dx*s + dy*c))
    p0 = Pk[f0]
    ring = np.searchsorted(SIDES/2, cheb(p0))            # the smallest map holding each particle now
    out = dict(stop={}, err={}, err_id={}, w=[], thr=[], ok=[])
    # 1. stopped-looking: share of probability moving under one cell during each map's lead
    for name, leads in CANDS.items():
        for j, ls in enumerate(leads):
            for m, L in enumerate(ls):
                mv = np.hypot(*(at(Pk, f0, L) - p0).T)
                inm = cheb(p0) <= SIDES[m]/2
                if inm.any():
                    out["stop"].setdefault(f"{m}:{L}", []).append((float(wk[inm & (mv < SIDES[m]/64)].sum()), float(wk[inm].sum())))
    # 2. identity-free vs with identities. The map shows the same particles now and after the lead, as counts per cell: the
    # particles inside the map now (up to N_MATCH, unweighted like the count channel) are snapped to the cells now and after the
    # lead, the labels dropped, and the two equal-size sets matched by the least total squared move (balanced assignment; an
    # unbalanced one let the matcher pick the future points lying on the present ones = "nothing moved", fixed 2026-11-19). C3 chains
    # the set after the first lead to the set after the second. Scored on the matched particles read on this map (ring), weighted.
    err = {n: [] for n in CANDS}; err_id = {n: [] for n in CANDS}; wts, thrs, taus = [], [], []
    for m in range(len(SIDES)):
        side = SIDES[m]
        U = np.flatnonzero(cheb(p0) <= side/2)
        if len(U) < 5:
            continue
        U = rng.choice(U, min(N_MATCH, len(U)), replace=False)
        E = (ring[U] == m) & ok[U]
        if not E.any():
            continue
        now_q = snap(p0[U], x, c, s, side)
        for name, leads in CANDS.items():
            futs = [at(Pk, f0, ls[m])[U] for ls in leads]
            fq = [snap(f, x, c, s, side) for f in futs]
            prev, chain = now_q, np.arange(len(U))            # chain: which point of the current set each particle holds
            pts_free = [now_q]
            for k, F in enumerate(fq):
                r_, c_ = linear_sum_assignment(((prev[:, None, :] - F[None, :, :])**2).sum(-1))
                to = np.empty(len(U), int); to[r_] = c_
                chain = to[chain]
                pts_free.append(F[chain]); prev = F
            times = np.array([0.0] + [ls[m] for ls in leads])
            for pts, sink in ((pts_free, err[name]), ([p0[U]] + futs, err_id[name])):
                Q = np.stack([p[E] for p in pts])

                def traj(kk, Q=Q):
                    t = kk*DT
                    if t <= times[-1]:
                        jj = min(np.searchsorted(times, t, side="right") - 1, len(times) - 2)
                        a = (t - times[jj])/(times[jj + 1] - times[jj])
                        return Q[jj]*(1 - a) + Q[jj + 1]*a
                    v = (Q[-1] - Q[-2])/(times[-1] - times[-2])
                    return Q[-1] + v*(t - times[-1])
                _, qe, oke = meet(traj, x, hd, n_tau)
                qe[~oke] = Q[-1][~oke]
                sink.append(np.hypot(*(qe - q[U][E]).T))
        qq = q[U][E]
        thrs.append(np.maximum(ERR_M, np.array([SIDES[min(np.searchsorted(SIDES/2, h), 6)]/64 for h in cheb(qq)])))
        wts.append(wk[U][E]); taus.append(k_true[U][E]*DT/60)
    if not wts:
        return None
    out["err"] = {n: np.concatenate(v) for n, v in err.items()}
    out["err_id"] = {n: np.concatenate(v) for n, v in err_id.items()}
    out["thr"], out["wt"], out["tau"] = np.concatenate(thrs), np.concatenate(wts), np.concatenate(taus)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", type=int, default=40)
    ap.add_argument("--steps", nargs="*", default=["2000000", "2400000", "2800000", "3200000", "3600000", "4000000"])
    a = ap.parse_args()
    from rl_search.scripts.real_map.vehicle_web import RUNS
    rids = [r for r in (f"eval-v2j-s3-k{k}-rl-r15_chungju-{s}" for s in a.steps for k in range(3)) if (RUNS/r/"run.json.gz").exists()]
    rng = np.random.default_rng(0)
    rows = {"flight": [], "pool": []}
    for src, P, f0, x, hd, w, act in itertools.chain(flight_scenes(rids), pool_scenes(a.pool)):
        o = scene(P, f0, x, hd, w, act, rng)
        if o is not None:
            rows[src].append(o)
    summary = {}
    TB = [0, 2, 5, 10, 20, 45]
    for src, rs in rows.items():
        st = {}
        for r in rs:
            for k, v in r["stop"].items():
                st.setdefault(k, []).extend(v)
        stop = {k: float(sum(a for a, _ in v)/max(sum(b for _, b in v), 1e-12)) for k, v in st.items()}
        thr = np.concatenate([r["thr"] for r in rs]); tau = np.concatenate([r["tau"] for r in rs])
        wt = np.concatenate([r["wt"]/r["wt"].sum() for r in rs])           # every scene weighs the same

        def share(bad, m=None):
            m = np.ones(len(bad), bool) if m is None else m
            return float(wt[m & bad].sum()/wt[m].sum()) if wt[m].sum() > 0 else None
        tb = np.digitize(tau, TB) - 1
        res = dict(scenes=len(rs), stopped=stop)
        for n in CANDS:
            ef = np.concatenate([r["err"][n] for r in rs]); ei = np.concatenate([r["err_id"][n] for r in rs])
            res[n] = dict(bad_free=share(ef > thr), bad_id=share(ei > thr),
                          free_by_tau=[share(ef > thr, tb == b) for b in range(5)], id_by_tau=[share(ei > thr, tb == b) for b in range(5)])
        res["C2 (칸 맞춤만)"] = dict(bad_free=0.0)      # snapping moves a point under 0.71 cell, never over the threshold (>= 1 cell)
        summary[src] = res
    (OUT/"lead_gap2.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    figure(summary)
    print(json.dumps(summary, ensure_ascii=False)[:3000])


def figure(S):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["AppleGothic", "sans-serif"]; plt.rcParams["axes.unicode_minus"] = False
    INK, MUTED, GRID, BG = "#1f1f1e", "#6b6b67", "#e4e3de", "#fcfcfb"
    fig, ax = plt.subplots(1, 2, figsize=(17, 6.4), facecolor=BG)
    names = ["1.6", "3.2", "6.4", "12.8", "25.6", "51.2", "102.4"]
    leads0 = CANDS["C0 지금"][0]; leads2 = CANDS["C3 두 장 (지금, 2배)"][1]
    for src, ls in (("flight", "-"), ("pool", "--")):
        st = S[src]["stopped"]
        y0 = [100*st.get(f"{m}:{L}", np.nan) for m, L in enumerate(leads0)]
        y2 = [100*st.get(f"{m}:{L}", np.nan) for m, L in enumerate(leads2)]
        lab = "비행" if src == "flight" else "학습 시나리오"
        ax[0].plot(names, y0, ls, marker="o", color="#6b6b67", label=f"지금 lead ({lab})")
        ax[0].plot(names, y2, ls, marker="o", color="#E69F00", label=f"2배 lead ({lab})")
    ax[0].set_title("lead 동안 한 칸도 못 움직여 '멈춘 것처럼' 보이는 확률 몫 (%)", loc="left", fontsize=11.5, color=INK)
    ax[0].set_xlabel("지도 한 변 (km)", color=MUTED); ax[0].legend(frameon=False, fontsize=9)
    labels = ["0~2분", "2~5분", "5~10분", "10~20분", "20~45분"]
    for src, ls in (("flight", "-"), ("pool", "--")):
        for n, col in (("C0 지금", "#6b6b67"), ("C3 두 장 (지금, 2배)", "#E69F00")):
            r = S[src][n]
            lab = "비행" if src == "flight" else "학습 시나리오"
            ax[1].plot(labels, [100*v if v is not None else np.nan for v in r["free_by_tau"]], ls, marker="o", color=col,
                       label=f"{n}: 입자 구분 없이 {100*r['bad_free']:.0f}% ({lab})")
            ax[1].plot(labels, [100*v if v is not None else np.nan for v in r["id_by_tau"]], ls, marker="x", color=col, alpha=.4,
                       label=f"{n}: 입자 구분 가정 {100*r['bad_id']:.0f}% ({lab})")
    ax[1].set_title("틀린 확률 몫 (%): 입자 구분 없이 지도 칸만으로 짐작 (진한 선) 대 구분 가정 (흐린 선)", loc="left", fontsize=11.5, color=INK)
    ax[1].legend(frameon=False, fontsize=8)
    for a_ in ax:
        a_.set_facecolor(BG); a_.grid(color=GRID, lw=.5); a_.tick_params(colors=MUTED)
        for sp in a_.spines.values():
            sp.set_color(GRID)
    fig.suptitle("lead 후보 (학습 없음) 2: 멈춘 것처럼 보이는 입자, 그리고 지도 그림만으로 짝을 맞춰 짐작할 때", x=.01, ha="left", fontsize=13, color=INK)
    fig.tight_layout()
    fig.savefig(OUT/"lead_gap2.png", dpi=78, facecolor=BG)


if __name__ == "__main__":
    main()

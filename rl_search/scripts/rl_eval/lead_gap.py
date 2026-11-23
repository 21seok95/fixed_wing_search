"""Which future channels do the observation maps need? A no-training check of the lead candidates (user 2026-11-19,
docs/scenario_redesign.md "lead 채널 후보와 학습 없이 고르는 방법").

Truth: for every particle of a scene, where the UAV meets it if it flies straight at it from where it is now (the first time the
particle is within the UAV's reach: distance / 33 m/s + the turn toward it, radius 143 m), and when.
Candidate estimate: what the map channels let one infer, optimistically (as if the network could tell the particles apart):
  C0  each map shows the particle now and after the map's lead (R15: 15 s ... 600 s); straight line through the two, then the same
      intercept rule on that line;
  C1  the same with lead = time to the map's edge (half side / 33 m/s: 24 s ... 26 min);
  C3  two leads (C0's and twice it): the polyline through the three, the last leg continued;
  C2  the intercept map itself: no estimate needed (only the cell size, the same for every candidate).
A particle is read on the smallest map that holds it now and at the candidate's future times (maps are squares around the UAV,
heading up); on no map: its position now (no future). Error = distance between the estimated and the true meeting point.

Scenes: R15's Jirisan validation flights (UAV pose from the flight, weights = the particles' remaining probability) and Chungju
training scenarios (pool files; the UAV at a launch point drawn as training draws it, and at 15, 30, 45 min 3-8 km from a particle,
heading random; equal weights).

    python -m rl_search.scripts.rl_eval.lead_gap --pool 40
Writes report/lead_gap.png, report/lead_gap_examples.png and report/lead_gap.json.
"""
from __future__ import annotations

import argparse
import base64
import gzip
import json
import pickle
from pathlib import Path

import numpy as np

OUT = Path("rl_search/data/real/amboy/rl_eval/report")
POOL = Path("rl_search/data/real/chungju120/rl_pool_v2")
V, R_TURN, DT = 33.3, 143.0, 10.0
SIDES = np.array([1600, 3200, 6400, 12800, 25600, 51200, 102400], float)
CANDS = {
    "C0 지금 (lead 15초~10분)": [[15, 30, 60, 150, 300, 600, 600]],
    "C1 lead = 지도 끝까지 가는 시간": [list(np.round(SIDES/2/V))],
    "C3 lead 두 장 (C0, 2배)": [[15, 30, 60, 150, 300, 600, 600], [30, 60, 120, 300, 600, 1200, 1200]],
    "C4 모든 지도 5초·10초": [[5]*7, [10]*7],                    # user 2026-11-19: from the 5 s decision step
    "C5 1.6 km 5·10초, 지도 2배마다 2배": [[5*2**k for k in range(7)], [10*2**k for k in range(7)]],
}
TAU_BINS = [0, 2, 5, 10, 20, 45]          # minutes
ERR_M = 500.0                             # about the camera footprint width


def wrap(a):
    return (a + np.pi) % (2*np.pi) - np.pi


def at(P, f0, lead_s):
    """Positions lead_s after frame f0 (linear between saved frames, last frame held)."""
    f = f0 + lead_s/DT
    a = int(np.floor(f)); b = min(a + 1, len(P) - 1); a = min(a, len(P) - 1)
    w = f - np.floor(f)
    return P[a]*(1 - w) + P[b]*w


def meet(traj_fn, x, hd, n_tau):
    """First tau (frames) at which the UAV at x, heading hd, reaches the trajectory; (frame, position, reached)."""
    taus = np.arange(n_tau)*DT
    best = np.full(traj_fn(0).shape[0], -1)
    pos = np.full((len(best), 2), np.nan)
    for k, tau in enumerate(taus):
        p = traj_fn(k)
        d = np.hypot(p[:, 0] - x[0], p[:, 1] - x[1])
        turn = R_TURN*np.abs(wrap(np.arctan2(p[:, 1] - x[1], p[:, 0] - x[0]) - hd))/V
        new = (best < 0) & (d/V + turn <= tau)
        best[new], pos[new] = k, p[new]
        if (best >= 0).all():
            break
    return best, pos, best >= 0


def scene(P, f0, x, hd, w, act):
    """Errors per candidate for one scene: P (T, N, 2) particle positions, f0 the scene frame, UAV x / hd, weights w, active mask."""
    n_tau = len(P) - f0
    keep = act & (w > 0)
    if keep.sum() < 10 or n_tau < 3:
        return None
    Pk = P[:, keep]
    k_true, q, ok = meet(lambda k: Pk[f0 + k], x, hd, n_tau)
    c, s = np.cos(hd), np.sin(hd)

    def cheb(p):                                     # half-side needed to hold p on a heading-up square around the UAV
        dx, dy = p[:, 0] - x[0], p[:, 1] - x[1]
        return np.maximum(np.abs(dx*c + dy*s), np.abs(-dx*s + dy*c))
    p0 = Pk[f0]
    hq = cheb(np.nan_to_num(q))
    cell = np.array([SIDES[min(np.searchsorted(SIDES/2, h), len(SIDES) - 1)]/64 for h in hq])
    out = dict(w=w[keep]/w.sum(), tau=k_true*DT/60, ok=ok, err={}, thr=np.maximum(ERR_M, cell), p0=p0, q=q,
               straight=np.hypot(*(q - p0).T)/np.maximum(_path_len(Pk, f0, k_true), 1.0))
    for name, leads in CANDS.items():
        est = np.full((len(p0), 2), np.nan)
        done = np.zeros(len(p0), bool)
        for m in range(len(SIDES)):
            Ls = [ls[m] for ls in leads]
            pts = [p0] + [at(Pk, f0, L) for L in Ls]
            inside = np.all([cheb(p) <= SIDES[m]/2 for p in pts], axis=0) & ~done
            if not inside.any():
                continue
            times = np.array([0.0] + Ls)
            Q = np.stack([p[inside] for p in pts])      # (k, n, 2)

            def traj(kk, Q=Q, times=times):
                t = kk*DT
                if t <= times[-1]:
                    j = min(np.searchsorted(times, t, side="right") - 1, len(times) - 2)
                    a = (t - times[j])/(times[j + 1] - times[j])
                    return Q[j]*(1 - a) + Q[j + 1]*a
                v = (Q[-1] - Q[-2])/(times[-1] - times[-2])
                return Q[-1] + v*(t - times[-1])
            _, qe, oke = meet(traj, x, hd, n_tau)
            qe[~oke] = Q[-1][~oke]
            est[inside], done[inside] = qe, True
        est[~done] = p0[~done]                          # on no map with its future: only where it is now
        out["err"][name] = np.hypot(*(est - q).T)
        out.setdefault("est", {})[name] = est
    return out


def _path_len(Pk, f0, k):
    L = np.zeros(Pk.shape[1])
    kmax = int(k.max()) if len(k) else 0
    for j in range(kmax):
        step = np.hypot(*(Pk[f0 + j + 1] - Pk[f0 + j]).T)
        L += np.where(j < k, step, 0.0)
    return L


def flight_scenes(rids):
    from rl_search.scripts.real_map.vehicle_web import RUNS
    for rid in rids:
        r = json.load(gzip.open(RUNS/rid/"run.json.gz", "rt"))
        o, res = r["opt"], r["res"]
        N = res["N"]
        P = np.frombuffer(base64.b64decode(res["pos"]), "<i2").reshape(res["T"], N, 2).astype(float)*res["pos_unit_m"]
        st = np.frombuffer(base64.b64decode(res["state"]), np.uint8).reshape(res["T"], N)
        W = np.frombuffer(base64.b64decode(o["weight"]), np.uint8).reshape(o["T"], N).astype(float)/255.0
        u = o["uavs"][0]
        t = np.frombuffer(base64.b64decode(u["log_t"]), np.float32).astype(float)
        xyy = np.frombuffer(base64.b64decode(u["log_xyyaw"]), np.float32).reshape(-1, 3).astype(float)
        for ts in np.arange(t[0] + 60, t[-1] - 300, 180.0):
            f0 = int(round(ts/DT))
            i = int(np.searchsorted(t, ts))
            if f0 >= min(o["T"], res["T"]) - 3:
                break
            yield "flight", P, f0, xyy[i, :2], xyy[i, 2], W[f0], st[f0] != 4


def pool_scenes(n, seed=0):
    from rl_search.rl_eval_set import V2, start_stage3
    from rl_search.rl_real import preload
    terr = preload("chungju120")["terr"]
    half = float(terr.axis[-1])
    files = sorted(POOL.glob("road_s*.pkl"))
    rng = np.random.default_rng(seed)
    for f in [files[i] for i in rng.choice(len(files), n, replace=False)]:
        sc = pickle.load(open(f, "rb"))["scen"]
        P = sc.track.astype(float)
        ex = np.asarray(sc.exit_step).astype(np.int64)
        L = np.asarray(sc.lkp, float)
        _, s, _ = start_stage3(L, rng, half, V2["ring_m"], terr=terr)
        x0 = np.array(s[:2])
        yield "pool", P, 0, x0, float(np.arctan2(*(L - x0)[::-1])), np.ones(P.shape[1]), (ex < 0) | (ex > 0)
        for m in (15, 30, 45):
            f0 = int(m*60/DT)
            if f0 >= len(P) - 3:
                break
            act = (ex < 0) | (ex > f0)
            j = rng.choice(np.flatnonzero(act))
            ang, dist = rng.uniform(0, 2*np.pi), rng.uniform(3000, 8000)
            x = P[f0, j] + dist*np.array([np.cos(ang), np.sin(ang)])
            yield "pool", P, f0, x, float(rng.uniform(-np.pi, np.pi)), np.ones(P.shape[1]), act


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", type=int, default=40)
    ap.add_argument("--steps", nargs="*", default=["2000000", "2400000", "2800000", "3200000", "3600000", "4000000"])
    a = ap.parse_args()
    rids = [f"eval-v2j-s3-k{k}-rl-r15_chungju-{s}" for s in a.steps for k in range(3)]
    from rl_search.scripts.real_map.vehicle_web import RUNS
    rids = [r for r in rids if (RUNS/r/"run.json.gz").exists()]
    rows = {"flight": [], "pool": []}
    examples = []
    import itertools
    for src, P, f0, x, hd, w, act in itertools.chain(flight_scenes(rids), pool_scenes(a.pool)):
        o = scene(P, f0, x, hd, w, act)
        if o is None:
            continue
        rows[src].append(o)
        e0 = o["err"][list(CANDS)[0]]
        okw = float(o["w"][o["ok"]].sum())
        bad = float(o["w"][o["ok"] & (e0 > o["thr"])].sum())/max(okw, 1e-12)
        mid = float(o["w"][o["ok"] & (o["tau"] >= 5) & (o["tau"] < 20)].sum())/max(okw, 1e-12)
        if src == "flight" and okw > 0.3 and mid > 0.5:             # flight scenes with most of the mass 5-20 min away, typical C0 error
            examples.append((abs(bad - 0.55), src, x, hd, o))
            examples = sorted(examples, key=lambda e: e[0])[:3]
    summary = figure(rows)
    figure_examples(examples)
    (OUT/"lead_gap.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print(json.dumps(summary, ensure_ascii=False)[:1500])


def _cat(rows):
    keys = ["w", "tau", "ok", "straight", "thr"]
    out = {k: np.concatenate([r[k] for r in rows]) for k in keys}
    out["w"] = np.concatenate([r["w"]/len(rows) for r in rows])     # every scene weighs the same
    out["err"] = {n: np.concatenate([r["err"][n] for r in rows]) for n in CANDS}
    return out


def figure(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["AppleGothic", "sans-serif"]; plt.rcParams["axes.unicode_minus"] = False
    INK, MUTED, GRID, BG = "#1f1f1e", "#6b6b67", "#e4e3de", "#fcfcfb"
    COL = dict(zip(CANDS, ["#6b6b67", "#2a78d6", "#E69F00", "#d6336c", "#009E73"]))
    fig, ax = plt.subplots(2, 2, figsize=(16, 10.5), facecolor=BG)
    summary = {}
    labels = [f"{a}~{b}분" for a, b in zip(TAU_BINS[:-1], TAU_BINS[1:])]
    for src, ls in (("flight", "-"), ("pool", "--")):
        if not rows[src]:
            continue
        D = _cat(rows[src])
        ok = D["ok"]
        summary[src] = dict(scenes=len(rows[src]), reached=float(D["w"][ok].sum()/D["w"].sum()))
        tb = np.digitize(D["tau"], TAU_BINS) - 1
        sb = np.digitize(D["straight"], [0.8, 0.95])            # 0 winding, 1 bent, 2 straight
        for n in CANDS:
            bad = ok & (D["err"][n] > D["thr"])
            share = [float(D["w"][bad & (tb == b)].sum()/max(D["w"][ok & (tb == b)].sum(), 1e-12)) for b in range(len(labels))]
            med = []
            for b in range(len(labels)):
                m = ok & (tb == b)
                if D["w"][m].sum() <= 0:
                    med.append(np.nan); continue
                o = np.argsort(D["err"][n][m]); cw = np.cumsum(D["w"][m][o])/D["w"][m].sum()
                med.append(float(D["err"][n][m][o][np.searchsorted(cw, .5)]/1000))
            by_s = [float(D["w"][bad & (sb == b)].sum()/max(D["w"][ok & (sb == b)].sum(), 1e-12)) for b in range(3)]
            total = float(D["w"][bad].sum()/D["w"][ok].sum())
            summary[src][n] = dict(bad_total=round(total, 3), bad_by_tau=np.round(share, 3).tolist(), median_km_by_tau=np.round(med, 2).tolist(),
                                   bad_by_shape=np.round(by_s, 3).tolist())
            lab = f"{n} ({'비행' if src == 'flight' else '학습 시나리오'})"
            ax[0, 0].plot(labels, np.array(share)*100, ls, marker="o", color=COL[n], label=lab)
            ax[0, 1].plot(labels, med, ls, marker="o", color=COL[n], label=lab)
            x = np.arange(3) + (0 if src == "flight" else 0.08)
            ax[1, 0].plot(x, np.array(by_s)*100, ls, marker="o", color=COL[n], label=lab)
        mass = [float(D["w"][ok & (tb == b)].sum()/D["w"][ok].sum()) for b in range(len(labels))]
        ax[1, 1].bar(np.arange(len(labels)) + (-0.2 if src == "flight" else 0.2), np.array(mass)*100, 0.38,
                     color="#2a78d6" if src == "flight" else "#9fc2ea", label="비행 (R15 지리산)" if src == "flight" else "학습 시나리오 (충주)")
        summary[src]["mass_by_tau"] = np.round(mass, 3).tolist()
    ax[0, 0].set_title(f"틀린 확률 몫: 오차 > {ERR_M:.0f} m이면서 그 지도 칸보다 큼 (만나는 데 걸리는 시간별)", loc="left", fontsize=11.5, color=INK)
    ax[0, 0].set_ylabel("%", color=MUTED)
    ax[0, 1].set_title("추정 오차 중앙값 (km)", loc="left", fontsize=11.5, color=INK); ax[0, 1].set_yscale("log")
    ax[1, 0].set_xticks(range(3), ["굽고 갈라진 길\n(곧은 거리/경로 < 0.8)", "조금 굽음\n(0.8~0.95)", "곧은 길\n(≥ 0.95)"])
    ax[1, 0].set_title("틀린 몫: 길 모양별 (만나는 곳까지 곧은 거리 ÷ 실제 경로)", loc="left", fontsize=11.5, color=INK); ax[1, 0].set_ylabel("%", color=MUTED)
    ax[1, 1].set_xticks(range(len(labels)), labels)
    ax[1, 1].set_title("확률이 어디에 있나: 만나는 데 걸리는 시간별 몫 (%)", loc="left", fontsize=11.5, color=INK)
    for a_ in ax.ravel():
        a_.set_facecolor(BG); a_.grid(color=GRID, lw=.5); a_.tick_params(colors=MUTED, labelsize=9)
        for sp in a_.spines.values():
            sp.set_color(GRID)
    ax[0, 0].legend(frameon=False, fontsize=8.5); ax[1, 1].legend(frameon=False, fontsize=9)
    fig.suptitle("lead 후보 비교 (학습 없음): 곧장 가면 만나는 위치를 각 후보의 채널로 얼마나 맞히나 (C2 도착 시각 지도는 정의상 0, 칸 크기만 남음)",
                 x=.01, ha="left", fontsize=13, color=INK)
    fig.tight_layout()
    fig.savefig(OUT/"lead_gap.png", dpi=78, facecolor=BG)
    return summary


def figure_examples(examples):
    import matplotlib.pyplot as plt
    INK, MUTED, BG = "#1f1f1e", "#6b6b67", "#fcfcfb"
    if not examples:
        return
    fig, axs = plt.subplots(1, len(examples), figsize=(7*len(examples), 7), facecolor=BG)
    axs = np.atleast_1d(axs)
    n0, n3 = list(CANDS)[0], list(CANDS)[2]
    rng = np.random.default_rng(0)
    for a_, (_, src, x, hd, o) in zip(axs, examples):
        ok = o["ok"]
        p0, q, e0 = o["p0"]/1000, o["q"]/1000, o["est"][n0]/1000
        bad = ok & (o["err"][n0] > o["thr"])
        a_.scatter(p0[:, 0], p0[:, 1], s=3, color="#9a9a94", lw=0, label="지금 입자", zorder=1)
        a_.scatter(q[ok, 0], q[ok, 1], s=3, color="#009E73", lw=0, label="곧장 가면 만나는 곳 (정답, C2가 보여 주는 것)", zorder=2)
        a_.scatter(e0[bad, 0], e0[bad, 1], s=3, color="#d6336c", lw=0, label="C0(지금 + lead 뒤)로 추정한 곳 (틀린 입자만)", zorder=3)
        sel = rng.choice(np.flatnonzero(bad), min(60, int(bad.sum())), replace=False) if bad.any() else []
        for i in sel:
            a_.plot([e0[i, 0], q[i, 0]], [e0[i, 1], q[i, 1]], color="#d6336c", lw=.4, alpha=.5, zorder=2)
        a_.plot(x[0]/1000, x[1]/1000, marker=(3, 0, np.degrees(hd) - 90), ms=15, color=INK, zorder=4)
        okw = o["w"][ok].sum()
        b0 = o["w"][bad].sum()/okw
        b3 = o["w"][ok & (o["err"][n3] > o["thr"])].sum()/okw
        a_.set_aspect("equal")
        a_.set_title(f"R15 지리산 비행 장면: 틀린 확률 C0 {b0*100:.0f}%, C3 {b3*100:.0f}%", loc="left", fontsize=11, color=INK)
        a_.legend(frameon=False, fontsize=8.5, loc="best", markerscale=4)
        a_.set_xlabel("동쪽 (km)", color=MUTED)
    fig.suptitle("예시 장면 (확률 대부분이 5~20분 거리): 초록 = 실제로 만나는 곳, 빨강 = 두 장으로 직선 추정한 곳, 선 = 같은 입자, 삼각형 = 기체",
                 x=.01, ha="left", fontsize=12.5, color=INK)
    fig.tight_layout()
    fig.savefig(OUT/"lead_gap_examples.png", dpi=78, facecolor=BG)


if __name__ == "__main__":
    main()

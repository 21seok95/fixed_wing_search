"""How much do the particles branch, in the training scenarios and on Chungju? (user 2026-11-18: is the circling / off-road searching
a shortage of branching cases in the training data, the Mojave road networks being simplified?)

Scenario sets, each drawn with its own generator (the same seeds rule as the pools, so they are the scenarios training saw):
  pool   - R10-R14 training (rl_eval_set.multi_training_scenario, targets that never stop, varied routes) on amboy / mojave 60 km
  chungju_rules - the same generator on the Chungju map (only the map differs)
  x2     - the redesigned scenarios on the stretched maps (rl_search.stretch, the next training)
  web    - the user's 15 Chungju web cases (web_cases_20261117.json)
Branches at 15 / 30 / 45 / 60 min: particles on a 500 m grid (in the env's metres), connected pieces (8 neighbours) holding 5 % of the
particles or more; and the effective number of pieces exp(entropy of the shares). Particles that left the map are left out unless the
scenario keeps them moving. Map structure: junctions (road nodes with 3+ roads), dead ends, road length, per 100 km^2 of the central
40 km (env metres, so a stretched map counts a quarter per area).

    python -m rl_search.scripts.rl_eval.branch_check run --n 30 --workers 8
    python -m rl_search.scripts.rl_eval.branch_check figure
"""
from __future__ import annotations

import argparse
import json
import warnings
from multiprocessing import get_context
from pathlib import Path

import numpy as np

OUT = Path("rl_search/data/real/amboy/rl_eval/report")
WORK = OUT/"branch_check"
TYPES = ("road", "mixed", "desert")
MINUTES = (15, 30, 45, 60)
SETS = {"pool": ("amboy", "mojave"), "chungju_rules": ("chungju",), "x2": ("amboy_x2", "mojave_x2"), "web": ("chungju",)}
SN = {"pool": "R10~R14 학습 풀 (사막 60 km)", "chungju_rules": "같은 생성 규칙, 충주 지도", "x2": "새 학습: 늘린 지도", "web": "웹 충주 사례 15개"}
_ST = {}


def branches(P, cell=500.0):
    from scipy import ndimage
    if len(P) == 0:
        return 0, 0.0
    q = np.floor((P - P.min(axis=0))/cell).astype(int)
    G = np.zeros(q.max(axis=0) + 1, int)
    np.add.at(G, (q[:, 0], q[:, 1]), 1)
    lab, n = ndimage.label(G > 0, structure=np.ones((3, 3)))
    if n == 0:
        return 0, 0.0
    s = ndimage.sum(G, lab, index=np.arange(1, n + 1))/len(P)
    big = int((s >= 0.05).sum())
    eff = float(np.exp(-(s*np.log(np.maximum(s, 1e-12))).sum()))
    return big, eff


def measure(scen):
    tr, ex, dt = scen.track, np.asarray(scen.exit_step), float(scen.save_dt_s)
    keep_out = bool(getattr(scen, "exits_move", False))
    out = {}
    L = np.asarray(scen.lkp, float)
    for m in MINUTES:
        k = min(int(round(m*60/dt)), tr.shape[0] - 1)
        P = tr[k].astype(float)
        live = np.ones(len(P), bool) if keep_out else ~((ex >= 0) & (ex <= k))
        b, e = branches(P[live])
        r95 = float(np.percentile(np.hypot(*(P[live] - L).T), 95)/1000) if live.any() else np.nan
        out[m] = dict(big=b, eff=e, r95=r95, left=float(1 - live.mean()))
    return out


def _init():
    from rl_search.real_scenario import VehicleScenarios
    from rl_search.rl_eval_set import vehicle_config
    from rl_search.rl_real import preload
    for r in ("amboy", "mojave", "chungju", "amboy_x2", "mojave_x2"):
        st = preload(r)
        _ST[r] = dict(st=st, vs=VehicleScenarios(st.get("base_sampler", st["sampler"]), vehicle_config(), region=st.get("base", r), net=st["net"]))


def job(a):
    warnings.filterwarnings("ignore")
    kind, region, typ, i = a
    st = _ST[region]
    if kind in ("pool", "chungju_rules"):
        from rl_search.rl_eval_set import multi_training_scenario
        base = 10_000_000 + {"amboy": 0, "mojave": 500_000, "chungju": 900_000}[region]
        seed = base + 1000*TYPES.index(typ) + i
        scen = multi_training_scenario(st["vs"], seed, typ, nostop=True, route=True)[0]
    elif kind == "x2":
        from rl_search.stretch import stretched_training_scenario
        seed = 10_000_000 + {"amboy_x2": 3_000_000, "mojave_x2": 3_500_000}[region] + 1000*TYPES.index(typ) + i
        scen = stretched_training_scenario(st["vs"], seed, typ, st["st"]["k"])[0]
    else:
        from dataclasses import replace
        from rl_search.vehicle import VehicleConfig
        c = json.loads(Path("rl_search/data/real/chungju/web_cases_20261117.json").read_text())[i]
        vc = VehicleConfig(**c["config"])
        vc = replace(vc, duration_s=max(float(vc.duration_s), 4200.0))
        from rl_search.real_scenario import VehicleScenarios
        vs = VehicleScenarios(st["st"]["sampler"], vc, region="chungju", net=st["st"]["net"])
        scen = vs.make_at(int(c["view"]["seed"]), c["view"]["lkp"], float(c["view"]["heading"]), c["view"].get("start"))
        typ = f"web{c['i']}"
    return dict(kind=kind, region=region, typ=typ, i=i, m={str(k): v for k, v in measure(scen).items()})


def map_stats(region):
    from rl_search.rl_real import preload
    st = preload(region)
    net, terr = st["net"], st["terr"]
    k = float(st.get("k", 1.0))
    xy = net.node_xy*k
    u, v = net.de_from[0::2], net.de_to[0::2]
    deg = np.bincount(np.concatenate([u, v]), minlength=len(xy))
    cen = (np.abs(xy) <= 20000*k).all(axis=1) if k == 1 else (np.abs(xy) <= 20000*k).all(axis=1)
    area = (40*k)**2/100.0                                  # 100 km^2 units of the central 40 km (env metres)
    mid = 0.5*(xy[u] + xy[v])
    ce = (np.abs(mid) <= 20000*k).all(axis=1)
    return dict(junctions=float((cen & (deg >= 3)).sum()/area), dead_ends=float((cen & (deg == 1)).sum()/area),
                road_km=float(net.edge_len[ce].sum()*k/1000/area), dead_share=float((cen & (deg == 1)).sum()/max((cen & (deg >= 1)).sum(), 1)))


def run(n, workers):
    warnings.filterwarnings("ignore")
    WORK.mkdir(parents=True, exist_ok=True)
    _init()
    jobs = [(kind, r, t, i) for kind in ("pool", "chungju_rules", "x2") for r in SETS[kind] for t in TYPES for i in range(n)]
    jobs += [("web", "chungju", "", i) for i in range(15)]
    with get_context("fork").Pool(workers) as pool:
        rows = []
        for j, r in enumerate(pool.imap_unordered(job, jobs, chunksize=1), 1):
            rows.append(r)
            if j % 25 == 0:
                print(f"{j}/{len(jobs)}", flush=True)
    maps = {r: map_stats(r) for r in ("amboy", "mojave", "chungju", "amboy_x2", "mojave_x2")}
    (WORK/"rows.json").write_text(json.dumps(dict(rows=rows, maps=maps)))
    print(json.dumps(maps, indent=1))


def figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["AppleGothic", "NanumGothic", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False
    INK, MUTED, GRID, BG = "#1f1f1e", "#6b6b67", "#e4e3de", "#fcfcfb"
    d = json.loads((WORK/"rows.json").read_text())
    rows, maps = d["rows"], d["maps"]
    KC = {"pool": "#2a78d6", "chungju_rules": "#d6336c", "x2": "#1baf7a", "web": "#eb6834"}
    fig, axs = plt.subplots(2, 3, figsize=(22, 12), facecolor=BG)
    # 1. big branches at 30 / 60 min by type and set
    for col, m in enumerate(("30", "60")):
        ax = axs[0, col]
        groups = [(k, t) for t in TYPES for k in ("pool", "chungju_rules", "x2")] + [("web", "")]
        pos = 0
        ticks = []
        for k, t in groups:
            v = [r["m"][m]["big"] for r in rows if r["kind"] == k and (t == "" or r["typ"] == t)]
            if not v:
                continue
            vals, cnt = np.unique(v, return_counts=True)
            ax.scatter(np.full(len(vals), pos), vals, s=20 + 300*cnt/len(v), color=KC[k], alpha=0.7)
            ax.plot([pos - 0.3, pos + 0.3], [np.mean(v)]*2, color=INK, lw=2)
            ticks.append((pos, ("" if not t else {"road": "도로형", "mixed": "혼합형", "desert": "사막형"}[t] + "\n") + SN[k][:9]))
            pos += 1
            if k == "x2":
                pos += 0.6
        ax.set_xticks([p for p, _ in ticks], [l for _, l in ticks], fontsize=7.5, rotation=0)
        ax.set_ylabel("갈래 수 (입자 5% 넘는 조각)", fontsize=9, color=MUTED)
        ax.set_title(f"① 신고 뒤 {m}분 갈래 수 (원 크기 = 그 값의 비율, 검은 선 = 평균)", loc="left", fontsize=11, color=INK)
    # 2. effective number over time
    ax = axs[0, 2]
    for k in ("pool", "chungju_rules", "x2", "web"):
        for t, ls in (("road", "-"), ("mixed", "--"), ("desert", ":")) if k != "web" else (("", "-"),):
            v = [[r["m"][str(mm)]["eff"] for mm in MINUTES] for r in rows if r["kind"] == k and (t == "" or r["typ"] == t)]
            if v:
                ax.plot(MINUTES, np.mean(v, axis=0), color=KC[k], ls=ls, lw=2, marker="o",
                        label=f"{SN[k]}" + (f" {dict(road='도로형', mixed='혼합형', desert='사막형')[t]}" if t else ""))
    ax.set_xlabel("신고 뒤 분", fontsize=9, color=MUTED); ax.set_ylabel("유효 갈래 수 (몫의 엔트로피)", fontsize=9, color=MUTED)
    ax.set_title("② 유효 갈래 수 (작은 조각까지 반영)", loc="left", fontsize=11, color=INK); ax.legend(fontsize=7, frameon=False)
    # 3. map structure
    ax = axs[1, 0]
    names = ["amboy", "mojave", "chungju", "amboy_x2", "mojave_x2"]
    lab = ["암보이 60", "바스토 60", "충주", "암보이 ×2", "바스토 ×2"]
    for j, (key, ttl) in enumerate((("junctions", "갈림길"), ("dead_ends", "막다른 끝"))):
        ax.bar(np.arange(5) + (j - 0.5)*0.38, [maps[n][key] for n in names], width=0.36, color=("#2a78d6", "#9a9a95")[j], label=ttl)
    ax.set_xticks(range(5), lab); ax.set_ylabel("100 km²당 개수 (학습 환경 거리 기준)", fontsize=9, color=MUTED)
    ax.set_title("③ 지도 구조: 중앙 40 km의 갈림길·막다른 끝 밀도", loc="left", fontsize=11, color=INK); ax.legend(fontsize=8.5, frameon=False)
    ax = axs[1, 1]
    ax.bar(range(5), [maps[n]["road_km"] for n in names], color="#1baf7a")
    ax.set_xticks(range(5), lab); ax.set_ylabel("100 km²당 도로 km", fontsize=9, color=MUTED)
    ax.set_title("④ 도로 밀도", loc="left", fontsize=11, color=INK)
    # 5. r95 vs branches at 60
    ax = axs[1, 2]
    for k in ("pool", "chungju_rules", "x2", "web"):
        v = [(r["m"]["60"]["r95"], r["m"]["60"]["big"]) for r in rows if r["kind"] == k]
        a = np.array(v, float)
        ax.scatter(a[:, 0], a[:, 1] + np.random.default_rng(0).uniform(-0.15, 0.15, len(a)), s=14, alpha=0.6, color=KC[k], label=SN[k])
    ax.set_xlabel("60분 95% 반경 (km, 학습 환경 거리)", fontsize=9, color=MUTED); ax.set_ylabel("60분 갈래 수", fontsize=9, color=MUTED)
    ax.set_title("⑤ 퍼진 정도와 갈래 수", loc="left", fontsize=11, color=INK); ax.legend(fontsize=8, frameon=False)
    for a_ in axs.ravel():
        a_.set_facecolor(BG); a_.grid(color=GRID, lw=.5); a_.tick_params(colors=MUTED, labelsize=8)
        for s in a_.spines.values():
            s.set_color(GRID)
    fig.suptitle("입자가 몇 갈래로 갈라지나: 학습 풀 대 충주 대 새 학습 시나리오 (500 m 칸의 이어진 조각 중 입자 5% 넘는 것)", x=0.01, ha="left", fontsize=13.5,
                 color=INK)
    fig.tight_layout()
    fig.savefig(OUT/"branch_check.png", dpi=80, facecolor=BG)
    L = ["# 갈래 수", "", "| 묶음 | 유형 | 30분 갈래 (평균) | 60분 갈래 (평균) | 60분 유효 갈래 | 60분 R95 km | 개수 |", "|---|---|---|---|---|---|---|"]
    for k in ("pool", "chungju_rules", "x2", "web"):
        for t in (TYPES if k != "web" else ("",)):
            v = [r for r in rows if r["kind"] == k and (t == "" or r["typ"] == t)]
            if v:
                L.append(f"| {SN[k]} | {t or '-'} | {np.mean([r['m']['30']['big'] for r in v]):.2f} | {np.mean([r['m']['60']['big'] for r in v]):.2f} | "
                         f"{np.mean([r['m']['60']['eff'] for r in v]):.2f} | {np.nanmean([r['m']['60']['r95'] for r in v]):.1f} | {len(v)} |")
    L += ["", "| 지도 | 갈림길/100 km² | 막다른 끝/100 km² | 막다른 끝 비율 | 도로 km/100 km² |", "|---|---|---|---|---|"]
    for n_, l_ in zip(names, lab):
        L.append(f"| {l_} | {maps[n_]['junctions']:.1f} | {maps[n_]['dead_ends']:.1f} | {maps[n_]['dead_share']:.2f} | {maps[n_]['road_km']:.0f} |")
    (OUT/"branch_check.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=("run", "figure"))
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    run(a.n, a.workers) if a.what == "run" else figure()


if __name__ == "__main__":
    main()

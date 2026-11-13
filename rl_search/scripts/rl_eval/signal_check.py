"""Learning-signal check (user 2026-11-12: road cases and the start of a flight give large rewards, desert cases and the end of a flight give
small observation values and small rewards, so does the policy learn less there?).

Flies a checkpoint on validation cases the way training does (sampled actions, the run's own settings from its config.json) and records at
every 5 s decision the reward, the critic's value and the level of the observation maps. From these it computes what one PPO update sees:
GAE advantages (gamma and lambda of the run), their size after the normalisation PPO applies to a mixed minibatch (pooled over all
decisions), the critic's error against the return that followed, and the map levels, by target type and by 10 min of flight time.

    python -m rl_search.scripts.rl_eval.signal_check --run amboy_r6 --per-type 3
Writes rl_search/data/real/amboy/rl_eval/report/signal_check_<run>_<step>.json and .png.
"""
from __future__ import annotations

import argparse
import json
import re
import warnings
from multiprocessing import get_context
from pathlib import Path

import numpy as np

OUT = Path("rl_search/data/real/amboy/rl_eval/report")
BINS = np.arange(0, 61, 10)
TYPES = ("road", "mixed", "desert")
TN = {"road": "도로형", "mixed": "혼합형", "desert": "사막형"}


def settings_of(c: dict) -> dict:
    """The env settings of a training run, built from its config.json the way train_rl builds them."""
    from rl_search.scripts.rl_eval.sarops_ref import DEFS
    a = c.get("area_def", "window")
    tau, dead, low = float(c.get("blur_tau", 0) or 0), float(c.get("blur_dead_roll", 0) or 0), float(c.get("blur_tau_low", 0) or 0)
    sg, so = float(c.get("area_sigma", 0) or 0), float(c.get("area_sigma_off", 0) or 0)
    return ({} if a == "window" else dict(DEFS[a])) | ({"blur_tau": tau} if tau else {}) \
        | ({"area_roll_max": float(c.get("area_roll_max", 20.0))} if a == "realtime_t" else {}) \
        | ({"area_w": float(c["pos_w"])} if float(c.get("pos_w", 1) or 1) != 1 else {}) \
        | ({"blur_dead_roll": dead} if tau and dead else {}) | ({"blur_tau_low": low} if tau and dead and low else {}) \
        | ({"area_sigma": sg} if sg and sg != 150 else {}) | ({"area_sigma_off": so} if so else {}) \
        | ({"handover_km": float(c["handover_km"])} if c.get("handover_km") else {}) \
        | ({"edge_mode": "none", "off_map_zero": True} if c.get("no_edge") else {}) \
        | ({"area_img_min": float(c["area_img_min"])} if c.get("area_img_min") else {}) \
        | ({"big_map_m": float(c["big_map_km"])*1000.0} if c.get("big_map_km") else {}) | ({"rel_maps": True} if c.get("rel_maps") else {}) \
        | ({"fp_rate": True} if c.get("fp_rate") else {}) | ({"a_weight": float(c["a_weight"])} if c.get("a_weight") is not None and float(c["a_weight"]) != 1.0 else {}) \
        | ({"pos_norm": True} if c.get("pos_norm") else {})


def fly(job) -> dict:
    """One case with sampled actions: per decision the minutes after the launch, reward, value, and map levels."""
    model_path, split, k, s, seed = job
    warnings.filterwarnings("ignore")
    import torch
    from rl_search.rl_eval_set import case_type
    from rl_search.rl_real_eval import _case, _env, _model
    torch.manual_seed(seed)
    case, scen, _, _ = _case(split, 3, k)
    model, env = _model(model_path), _env(s)
    obs, _ = env.reset(options=case.options(scen))
    names = [m[0] for m in env.cfg.maps]
    t_l = float(case.t_launch_s)
    rows, term, trunc = [], False, False
    while not (term or trunc):
        ot, _ = model.policy.obs_to_tensor(obs)
        with torch.no_grad():
            v = float(model.policy.predict_values(ot).cpu().numpy().ravel()[0])
        lv = []
        for n in names:                                        # channel 0 = remaining POS now, 2 = A value
            o = obs[n]
            lv += [float(o[0].max()), float(o[0].mean()), float(o[2].max()), float(o[2].mean())]
        a, _ = model.predict(obs, deterministic=False)
        t = (env.env.t - t_l)/60.0
        obs, r, term, trunc, info = env.step(a)
        rows.append([t, float(r), v, float(obs["vec"][5])] + lv)
    return dict(k=k, typ=case_type(split, k), names=names, terminated=bool(term), rows=rows)


def gae(r, v, gamma, lam, terminated=True):
    adv = np.zeros_like(r)
    last = 0.0
    for i in range(len(r) - 1, -1, -1):
        nv = 0.0 if i == len(r) - 1 and terminated else (v[i + 1] if i + 1 < len(r) else v[i])
        d = r[i] + gamma*nv - v[i]
        last = d + gamma*lam*last
        adv[i] = last
    g = np.zeros_like(r)
    acc = 0.0
    for i in range(len(r) - 1, -1, -1):
        acc = r[i] + gamma*acc
        g[i] = acc
    return adv, g


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="amboy_r6", help="training run under rl_search/data/rl_real")
    ap.add_argument("--model", type=Path, default=None, help="checkpoint (default: the run's newest ckpt/model_<step>.zip)")
    ap.add_argument("--split", default="mval")
    ap.add_argument("--per-type", type=int, default=3, help="validation cases per target type")
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    run = Path("rl_search/data/rl_real")/args.run
    c = json.loads((run/"config.json").read_text())
    s = settings_of(c)
    model = args.model or max(run.glob("ckpt/model_*.zip"), key=lambda p: int(re.findall(r"model_(\d+)", p.name)[0]))
    step = int(re.findall(r"(\d+)", model.name)[0])
    from rl_search.rl_real import preload
    preload("amboy")
    ks = [k for k in range(3*args.per_type)]
    with get_context("fork").Pool(args.workers) as pool:
        flights = pool.map(fly, [(str(model), args.split, k, s, 1000 + k) for k in ks], chunksize=1)
    gamma, lam = float(c.get("gamma", 0.995)), float(c.get("gae_lambda", 0.95))
    allr = []
    for f in flights:
        a = np.array(f["rows"])
        adv, g = gae(a[:, 1], a[:, 2], gamma, lam, f["terminated"])
        f["adv"], f["ret"] = adv, g
        allr.append(np.column_stack([a, adv, g]))
    pooled = float(np.concatenate([f["adv"] for f in flights]).std())
    names = flights[0]["names"]
    nm = len(names)
    # columns: t, r, v, mass, (pos max, pos mean, A max, A mean) per map, adv, ret
    summ = {}
    for typ in TYPES:
        A = np.concatenate([x for x, f in zip(allr, flights) if f["typ"] == typ])
        adv, ret, v = A[:, -2], A[:, -1], A[:, 2]
        for lo, hi in zip(BINS[:-1], BINS[1:]):
            m = (A[:, 0] >= lo) & (A[:, 0] < hi)
            if m.sum() < 10:
                continue
            row = dict(n=int(m.sum()), reward=float(A[m, 1].mean()), weight=float(np.abs(adv[m]).mean()/pooled),
                       crit_err=float(np.abs(ret[m] - v[m]).mean()/max(np.abs(ret[m]).mean(), 1e-9)), ret=float(ret[m].mean()),
                       mass=float(A[m, 3].mean()))
            for i, n in enumerate(names):
                row[f"{n}_pos_max"], row[f"{n}_pos_mean"] = float(A[m, 4 + 4*i].mean()), float(A[m, 5 + 4*i].mean())
                row[f"{n}_a_max"], row[f"{n}_a_mean"] = float(A[m, 6 + 4*i].mean()), float(A[m, 7 + 4*i].mean())
            summ[f"{typ}:{lo}-{hi}"] = row
    OUT.mkdir(parents=True, exist_ok=True)
    tag = f"signal_check_{args.run}_{step}"
    (OUT/f"{tag}.json").write_text(json.dumps(dict(model=str(model), settings=s, gamma=gamma, lam=lam, pooled_adv_std=pooled, names=names,
                                                   summary=summ, flights=[dict(k=f["k"], typ=f["typ"], rows=f["rows"]) for f in flights]), indent=0))
    print(f"model {model} ({len(flights)} flights, {sum(len(f['rows']) for f in flights)} decisions), pooled advantage SD {pooled:.3f}")
    print(f"{'type':7s} {'min':6s} {'n':>4s} {'reward':>7s} {'weight':>7s} {'critic err':>10s} {'return':>7s} {'POS max 1.6':>11s} {'POS max 6.4':>11s} "
          f"{'A max 1.6':>9s} {'mass':>5s}")
    for key, r in summ.items():
        typ, b = key.split(":")
        print(f"{typ:7s} {b:6s} {r['n']:4d} {r['reward']:7.3f} {r['weight']:7.2f} {r['crit_err']:10.2f} {r['ret']:7.1f} "
              f"{r[names[0] + '_pos_max']:11.2f} {r[names[1] + '_pos_max']:11.2f} {r[names[0] + '_a_max']:9.2f} {r['mass']:5.2f}")
    figure(summ, names, OUT/f"{tag}.png", f"{args.run} {step:,} 스텝 · 검증 {len(flights)}개 비행(유형당 {args.per_type}) · 학습처럼 행동을 뽑아서")
    print("saved", OUT/f"{tag}.png")


def figure(summ, names, out, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["AppleGothic", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False
    INK, MUTED, GRID, BG = "#1f1f1e", "#6b6b67", "#e4e3de", "#fcfcfb"
    TC = {"road": "#2a78d6", "mixed": "#eb6834", "desert": "#1baf7a"}
    panels = [("reward", "(가) 5초 결정당 보상"), ("weight", "(나) 정책을 움직이는 힘: |어드밴티지| ÷ 전체 표준편차"),
              ("crit_err", "(다) 가치 함수 오차 ÷ 실제로 받은 남은 보상"), (f"{names[0]}_pos_max", "(라) 1.6 km 지도 POS 채널 최댓값"),
              (f"{names[1]}_pos_max", "(마) 6.4 km 지도 POS 채널 최댓값"), (f"{names[0]}_a_max", "(바) 1.6 km 지도 A 채널 최댓값")]
    fig, axs = plt.subplots(2, 3, figsize=(15, 8.6), facecolor=BG)
    mids = (BINS[:-1] + BINS[1:])/2
    for ax, (key, ttl) in zip(axs.ravel(), panels):
        ax.set_facecolor(BG)
        ax.grid(color=GRID, lw=.6)
        for sp in ax.spines.values():
            sp.set_color(GRID)
        for typ in TYPES:
            xs, ys = [], []
            for lo, hi, mid in zip(BINS[:-1], BINS[1:], mids):
                r = summ.get(f"{typ}:{lo}-{hi}")
                if r is not None:
                    xs.append(mid)
                    ys.append(r[key])
            ax.plot(xs, ys, color=TC[typ], lw=2, marker="o", ms=5, label=TN[typ])
        ax.set_title(ttl, loc="left", color=INK, fontsize=10.5)
        ax.set_xticks(BINS)
        ax.tick_params(colors=MUTED, labelsize=9)
        ax.set_xlabel("이륙 뒤 (분)", color=MUTED, fontsize=9)
        if key == "weight":
            ax.axhline(1.0, color=MUTED, lw=1, ls="--")
    axs[0, 0].legend(fontsize=9, frameon=False)
    fig.suptitle(title, x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=120, facecolor=BG)


if __name__ == "__main__":
    main()

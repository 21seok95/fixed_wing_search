"""Does the policy tell the target types apart, and does it fly them differently? (user 2026-11-16, after R12; docs/rl_experiments.md)

1. Flies the validation cases (nval + bval, 30 flights) with a checkpoint exactly as eval_rl does (deterministic) and records at every
   5 s decision: the shared features (joint layer, 256), the policy-head latent (128), simple summary statistics of the observation
   (per map: particle mass, share of occupied cells, spread, absolute POS, lead occupancy; plus the vector), the action, POS and the
   remaining belief. Cached in <out>/type_views_<label>.npz.
2. Type probe: a multinomial logistic regression predicts the type (road / mixed / desert) from each feature set, trained on 24 flights
   and tested on the 6 held-out flights (5 folds, 2 flights of each type per fold, so no flight is in its own training set). Accuracy by
   minutes after launch; chance is 1/3. If the features know the type but the flights do not differ, recognition is not the problem.
3. Behaviour by type and phase: action size, share of decisions turning faster than 1.49 deg/s, turn-sign runs, POS over time, and from
   circle_check's per-minute file (circle_<label>.json, if present): circling share, POS per minute circling / other, belief near / far.

    python -m rl_search.scripts.rl_eval.type_views --model rl_search/data/rl_real/r12_2map/ckpt/model_6000000.zip --label r12_2map-6000000
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np

TYPES = ("road", "mixed", "desert")
TN = {"road": "도로형", "mixed": "혼합형", "desert": "사막형"}
TC = {"road": "#2a78d6", "mixed": "#eb6834", "desert": "#1baf7a"}
PHASES = ((0, 15), (15, 30), (30, 45), (45, 90))
PH_NAMES = ("0~15분", "15~30분", "30~45분", "45분~")


def obs_stats(obs: dict) -> np.ndarray:
    out = []
    for name in sorted(k for k in obs if k != "vec"):
        o = np.asarray(obs[name], float)
        c0 = o[0]
        n = c0.shape[0]
        yy, xx = np.mgrid[0:n, 0:n]
        m = c0.sum()
        if m > 0:
            cx, cy = (c0*xx).sum()/m, (c0*yy).sum()/m
            spread = np.sqrt((c0*((xx - cx)**2 + (yy - cy)**2)).sum()/m)
        else:
            spread = 0.0
        out += [c0.mean(), (c0 > 0).mean(), spread, o[3].mean(), (o[1] > 0).mean()]
    return np.r_[out, np.asarray(obs["vec"], float)]


def fly(model_path: str, splits, out_npz: Path):
    import torch
    from rl_search.rl_eval_set import case_type
    from rl_search.rl_real_eval import _case, _env, _model
    from rl_search.scripts.rl_eval.signal_check import settings_of
    run = Path(model_path).parent.parent if Path(model_path).parent.name == "ckpt" else Path(model_path).parent
    settings = settings_of(json.loads((run/"config.json").read_text()))
    model = _model(model_path)
    pol = model.policy
    rec = {k: [] for k in ("split", "k", "typ", "minute", "a", "pos", "rem", "feat", "lat", "stat")}
    for sp in splits:
        for k in range(15):
            t0 = time.perf_counter()
            case, scen, vres, hold = _case(sp, 3, k)
            env = _env(settings, case.region)
            e = env.env
            obs, info = env.reset(options=case.options(scen))
            term = trunc = False
            n = 0
            while not (term or trunc):
                a, _ = model.predict(obs, deterministic=True)
                with torch.no_grad():
                    ot, _ = pol.obs_to_tensor(obs)
                    f = pol.extract_features(ot)
                    lat, _ = pol.mlp_extractor(f)
                rec["split"].append(sp); rec["k"].append(k); rec["typ"].append(case_type(sp, k))
                rec["minute"].append((e.t - float(case.t_launch_s))/60.0)
                rec["a"].append(float(np.ravel(a)[0])); rec["feat"].append(f.numpy()[0]); rec["lat"].append(lat.numpy()[0])
                rec["stat"].append(obs_stats(obs))
                obs, r, term, trunc, info = env.step(a)
                rec["pos"].append(float(info["POS"])); rec["rem"].append(float(e.belief.w.sum()))
                n += 1
            print(f"{sp} k{k} {case_type(sp, k)}: {n} decisions, POS {info['POS']:.3f}, {time.perf_counter() - t0:.0f} s", flush=True)
    arr = {k: np.asarray(v) for k, v in rec.items()}
    np.savez_compressed(out_npz, **arr)
    return arr


def fit_predict(Xtr, ytr, Xte, l2=1e-2, iters=300):
    import torch
    import torch.nn.functional as F
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    xt = torch.tensor((Xtr - mu)/sd, dtype=torch.float32)
    yt = torch.tensor(ytr)
    W = torch.zeros(xt.shape[1], 3, requires_grad=True)
    b = torch.zeros(3, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], max_iter=iters, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(xt @ W + b, yt) + l2*(W**2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    with torch.no_grad():
        return (torch.tensor((Xte - mu)/sd, dtype=torch.float32) @ W + b).argmax(1).numpy()


def probe(arr, key):
    y = np.array([TYPES.index(t) for t in arr["typ"]])
    flights = sorted({(t, s, k) for t, s, k in zip(arr["typ"], arr["split"], arr["k"])}, key=lambda f: (TYPES.index(f[0]), f[1], f[2]))
    fold = {}
    for t in TYPES:
        for i, f in enumerate([f for f in flights if f[0] == t]):
            fold[f] = i % 5
    fid = np.array([fold[(t, s, k)] for t, s, k in zip(arr["typ"], arr["split"], arr["k"])])
    pred = np.zeros_like(y)
    X = arr[key].astype(float)
    for f in range(5):
        tr, te = fid != f, fid == f
        pred[te] = fit_predict(X[tr], y[tr], X[te])
    return y, pred


def main():
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True, help="name of the outputs (and of circle_check's circle_<label>.json)")
    ap.add_argument("--splits", nargs="+", default=["nval", "bval"])
    ap.add_argument("--out", type=Path, default=Path("rl_search/data/real/amboy/rl_eval/report"))
    ap.add_argument("--circle", type=Path, default=None, help="circle_check per-minute file (default <out>/circle_<label>.json; "
                                                                  "R12 6M on the Mac: report/r12_circle_minutes.json)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    npz = args.out/f"type_views_{args.label}.npz"
    arr = dict(np.load(npz, allow_pickle=False)) if npz.exists() else fly(args.model, args.splits, npz)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["AppleGothic", "NanumGothic", "Noto Sans CJK KR", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False
    INK, MUTED, GRID, BG = "#1f1f1e", "#6b6b67", "#e4e3de", "#fcfcfb"
    minute = arr["minute"]
    ph = np.digitize(minute, [p[1] for p in PHASES[:-1]])
    L = [f"# 유형 구분과 유형별 행동: {args.label} (검증 {len(set(zip(arr['split'], arr['k'])))}개 비행, 결정 {len(minute):,}개)", ""]
    # ---- probe
    sets = [("feat", f"합치기 층 {arr['feat'].shape[1]}개 값", "#1f1f1e"), ("lat", f"정책 머리 안쪽 {arr['lat'].shape[1]}개 값", "#d6336c"),
            ("stat", f"관측 요약 통계 {arr['stat'].shape[1]}개 (비교용)", "#9a9a95")]
    res = {}
    for key, name, _ in sets:
        y, p = probe(arr, key)
        res[key] = (y, p)
    fig, axs = plt.subplots(1, 3, figsize=(20, 5.6), facecolor=BG, gridspec_kw=dict(width_ratios=[1.2, 1, 1]))
    ax = axs[0]
    x = np.arange(len(PHASES))
    wbar = 0.8/len(sets)
    L += ["## 유형 맞히기 (본 적 없는 비행에서, 우연 = 0.33)", "", "| 특징 | 전체 | " + " | ".join(PH_NAMES) + " |", "|---|---|" + "---|"*len(PHASES)]
    for j, (key, name, col) in enumerate(sets):
        y, p = res[key]
        acc = [np.mean(p[ph == i] == y[ph == i]) if (ph == i).any() else np.nan for i in range(len(PHASES))]
        b = ax.bar(x + (j - 1)*wbar, acc, wbar*0.92, color=col, label=name)
        for bb, v in zip(b, acc):
            ax.text(bb.get_x() + bb.get_width()/2, v + 0.01, f"{v:.2f}", ha="center", fontsize=8.5, color=INK)
        L.append(f"| {name} | {np.mean(p == y):.2f} | " + " | ".join(f"{v:.2f}" for v in acc) + " |")
    ax.axhline(1/3, color=MUTED, ls="--", lw=1)
    ax.text(len(PHASES) - 0.5, 1/3 + 0.01, "우연 0.33", fontsize=8.5, color=MUTED, ha="right")
    ax.set_xticks(x); ax.set_xticklabels(PH_NAMES, fontsize=9.5, color=INK); ax.set_ylim(0, 1.25)
    ax.set_title("(가) 유형을 맞히는 비율 (이륙 뒤 시간대별, 본 적 없는 비행 6개씩 5번)", loc="left", fontsize=10.5, color=INK)
    ax.legend(fontsize=8.5, frameon=False, loc="upper center", ncol=3)
    counts = [int((ph == i).sum()) for i in range(len(PHASES))]
    L += ["", "시간대별 결정 수: " + ", ".join(f"{n} {c:,}" for n, c in zip(PH_NAMES, counts)), ""]
    for ax, key in zip(axs[1:], ("feat", "stat")):
        y, p = res[key]
        cm = np.array([[np.sum((y == i) & (p == j)) for j in range(3)] for i in range(3)], float)
        cm /= cm.sum(1, keepdims=True)
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=11, color="white" if cm[i, j] > 0.55 else INK)
        ax.set_xticks(range(3)); ax.set_xticklabels([TN[t] for t in TYPES]); ax.set_yticks(range(3)); ax.set_yticklabels([TN[t] for t in TYPES])
        ax.set_xlabel("맞힌 유형", fontsize=9, color=MUTED); ax.set_ylabel("실제 유형", fontsize=9, color=MUTED)
        ax.set_title(f"({'나' if key == 'feat' else '다'}) {dict((k, n) for k, n, _ in sets)[key]}: 실제 → 맞힌 유형 (행 합 1)", loc="left", fontsize=10.5, color=INK)
        L += [f"### {dict((k, n) for k, n, _ in sets)[key]}: 실제(행) → 맞힌 유형(열)", "", "| | " + " | ".join(TN[t] for t in TYPES) + " |", "|---|---|---|---|"]
        L += [f"| {TN[TYPES[i]]} | " + " | ".join(f"{cm[i, j]:.2f}" for j in range(3)) + " |" for i in range(3)] + [""]
    for ax in axs:
        ax.set_facecolor(BG); ax.tick_params(colors=MUTED, labelsize=8.5)
        for s in ax.spines.values():
            s.set_color(GRID)
    fig.suptitle(f"{args.label}: 모델이 유형(도로·혼합·사막)을 알아보는가", x=0.01, ha="left", fontsize=13, color=INK)
    fig.tight_layout()
    f1 = args.out/f"type_probe_{args.label}.png"
    fig.savefig(f1, dpi=100, facecolor=BG)
    # ---- behaviour by type and phase
    a = arr["a"]
    typ = arr["typ"]
    fast = np.abs(a)*49.0/5.0 > 1.49
    flight = np.array([f"{s}{k}" for s, k in zip(arr["split"], arr["k"])])
    same = np.r_[False, (np.sign(a[1:]) == np.sign(a[:-1])) & (flight[1:] == flight[:-1]) & (np.abs(a[1:]) > 0.05) & (np.abs(a[:-1]) > 0.05)]
    fig, axs = plt.subplots(1, 4, figsize=(22, 5.6), facecolor=BG)
    L += ["## 유형별 행동 (같은 비행 기록)", "", "| 유형 | 시간대 | 결정 수 | 평균 \\|a\\| | 1.49°/s보다 빠른 선회 몫 | 같은 방향 연속 선회 몫 |", "|---|---|---|---|---|---|"]
    wbar = 0.8/3
    for j, t in enumerate(TYPES):
        m_abs, m_fast = [], []
        for i in range(len(PHASES)):
            s = (typ == t) & (ph == i)
            m_abs.append(np.abs(a[s]).mean() if s.any() else np.nan)
            m_fast.append(fast[s].mean() if s.any() else np.nan)
            L.append(f"| {TN[t]} | {PH_NAMES[i]} | {int(s.sum()):,} | {m_abs[-1]:.3f} | {m_fast[-1]:.2f} | {same[s].mean() if s.any() else np.nan:.2f} |")
        axs[0].bar(x + (j - 1)*wbar, m_fast, wbar*0.92, color=TC[t], label=TN[t])
    axs[0].set_title("(가) 1.49°/s보다 빨리 도는 결정의 몫", loc="left", fontsize=10.5, color=INK)
    axs[0].set_xticks(x); axs[0].set_xticklabels(PH_NAMES, fontsize=9.5); axs[0].legend(fontsize=8.5, frameon=False); axs[0].set_ylim(0, 1)
    # POS and remaining belief over time
    grid = np.arange(0, 66, 1.0)
    for t in TYPES:
        curves = []
        for fl in sorted(set(flight[typ == t])):
            s = flight == fl
            curves.append(np.interp(grid, minute[s], arr["pos"][s], left=np.nan, right=arr["pos"][s][-1]))
        c = np.nanmean(np.array(curves), 0)
        axs[1].plot(grid, c, color=TC[t], lw=2, label=TN[t])
    axs[1].set_title("(나) 평균 POS (이륙 뒤 분, 정책이 넘겨받은 뒤부터)", loc="left", fontsize=10.5, color=INK)
    axs[1].set_xlabel("이륙 뒤 분", fontsize=9, color=MUTED); axs[1].legend(fontsize=8.5, frameon=False); axs[1].grid(color=GRID, lw=.6)
    circ = args.circle or args.out/f"circle_{args.label}.json"
    if circ.exists():
        rows = json.loads(circ.read_text())
        L += ["", f"## 원 선회 (circle_check 분당 기록, {len(rows):,}분, 앞뒤 3분 한 방향 330° 이상)", "",
              "| 유형 | 시간대 | 분 수 | 원 선회 몫 | 분당 POS 원 선회 / 나머지 | 원 선회 때 3 km 안 / 3~15 km 남은 확률 |", "|---|---|---|---|---|---|"]
        for j, t in enumerate(TYPES):
            share, g_c, g_o = [], [], []
            for i, (lo, hi) in enumerate(PHASES):
                rr = [r for r in rows if r["typ"] == t and lo <= r["m"] < hi]
                cc = [r for r in rr if r["net_turn"] >= 330]
                oo = [r for r in rr if r["net_turn"] < 330]
                f = lambda L_, k: np.mean([r[k] for r in L_]) if L_ else np.nan
                share.append(len(cc)/max(len(rr), 1) if rr else np.nan)
                g_c.append(f(cc, "gain")); g_o.append(f(oo, "gain"))
                L.append(f"| {TN[t]} | {PH_NAMES[i]} | {len(rr)} | {share[-1]:.2f} | {g_c[-1]:.4f} / {g_o[-1]:.4f} | {f(cc, 'near3'):.3f} / {f(cc, 'far'):.3f} |")
            axs[2].bar(x + (j - 1)*wbar, share, wbar*0.92, color=TC[t], label=TN[t])
            axs[3].plot(x, g_o, color=TC[t], lw=2, marker="o", label=f"{TN[t]} 나머지")
            axs[3].plot(x, g_c, color=TC[t], lw=1.5, ls="--", marker="s", label=f"{TN[t]} 원 선회 중")
        axs[2].set_title("(다) 원 선회한 분의 몫", loc="left", fontsize=10.5, color=INK)
        axs[2].set_xticks(x); axs[2].set_xticklabels(PH_NAMES, fontsize=9.5); axs[2].legend(fontsize=8.5, frameon=False); axs[2].set_ylim(0, 1)
        axs[3].set_title("(라) 분당 POS 증가: 실선 = 원 선회 아닌 분, 점선 = 원 선회 중", loc="left", fontsize=10.5, color=INK)
        from matplotlib.ticker import FuncFormatter
        axs[3].set_xticks(x); axs[3].set_xticklabels(PH_NAMES, fontsize=9.5); axs[3].set_yscale("log"); axs[3].legend(fontsize=7.5, frameon=False, ncol=2)
        axs[3].yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        axs[3].grid(color=GRID, lw=.6)
    for ax in axs:
        ax.set_facecolor(BG); ax.tick_params(colors=MUTED, labelsize=8.5)
        for s in ax.spines.values():
            s.set_color(GRID)
    fig.suptitle(f"{args.label}: 유형별·시간대별 행동", x=0.01, ha="left", fontsize=13, color=INK)
    fig.tight_layout()
    f2 = args.out/f"type_behavior_{args.label}.png"
    fig.savefig(f2, dpi=100, facecolor=BG)
    (args.out/f"type_views_{args.label}.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print("saved", f1, f2, args.out/f"type_views_{args.label}.md")


if __name__ == "__main__":
    main()

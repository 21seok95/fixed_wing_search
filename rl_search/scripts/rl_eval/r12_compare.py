"""R12 (r12_2map: R10 two maps from scratch with minibatch 770 and target_kl 0.03) against R10 two maps (r10_2map) at the same steps:
validation ratio to SAROPS overall, by split and by target type, the training episodes (TensorBoard, hold-out POS of the last 100
episodes), the share of PPO updates cut short by the KL cap, and a paired per-case comparison over the evaluations both runs have.

    python -m rl_search.scripts.rl_eval.r12_compare --out rl_search/data/real/amboy/rl_eval/report
Writes <out>/r12_vs_r10.png and .md.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rl_search.scripts.rl_eval.r11_compare import SC, TC, TN, ppo_log
from rl_search.scripts.rl_eval.train_report import INK, MUTED, korean_font, style

C_NEW = "#2a78d6"


def tb_all(tb: Path, tag: str, k: int = 5):
    """One scalar over every TensorBoard run of the folder (a resumed run writes a new ppo_<n>), running mean over k rollouts."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    pts = {}
    for d in sorted(tb.glob("ppo_*")):
        ea = EventAccumulator(str(d), size_guidance={"scalars": 0})
        ea.Reload()
        if tag in ea.Tags()["scalars"]:
            pts.update({e.step: e.value for e in ea.Scalars(tag)})
    x = np.array(sorted(pts))
    y = np.array([pts[s] for s in x])
    if len(y) > k:
        y = np.convolve(y, np.ones(k)/k, mode="valid")
        x = x[k - 1:]
    return x/1e6, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="r10_2map")
    ap.add_argument("--run", default="r12_2map")
    ap.add_argument("--runs", type=Path, default=Path("rl_search/data/rl_real"))
    ap.add_argument("--logs", nargs=2, default=["r10_2map_train.log", "r12_train.log"], help="training logs of base and run (in --runs)")
    ap.add_argument("--out", type=Path, default=Path("rl_search/data/real/amboy/rl_eval/report"))
    args = ap.parse_args()
    korean_font()
    base = {e["step"]: e for e in json.loads((args.runs/args.base/"state.json").read_text())["history"]["3"]}
    new = {e["step"]: e for e in json.loads((args.runs/args.run/"state.json").read_text())["history"]["3"]}
    last = max(new)
    base = {s: e for s, e in base.items() if s <= last + 400_000}
    both = [s for s in sorted(new) if s in base]

    lines = [f"{args.run} (R12, 처음부터 미니배치 770·KL 0.03) 대 {args.base} (R10 두 지도), 같은 스텝", "",
             "| 스텝 | R10 전체 | R12 전체 | R10 nval | R12 nval | R10 bval | R12 bval | "
             + " | ".join(f"R10 {TN[t]} | R12 {TN[t]}" for t in TN) + " |", "|" + "---|"*13]
    f = lambda e, g: f"{g(e):.3f}" if e else "-"
    for s in sorted(set(base) | set(new)):
        b, r = base.get(s), new.get(s)
        lines.append(f"| {s/1e6:.1f}M | {f(b, lambda e: e['ratio'])} | {f(r, lambda e: e['ratio'])} | "
                     + " | ".join(f"{f(b, lambda e, k=k: e['by_split'][k])} | {f(r, lambda e, k=k: e['by_split'][k])}" for k in SC) + " | "
                     + " | ".join(f"{f(b, lambda e, t=t: e['type_ratio'][t])} | {f(r, lambda e, t=t: e['type_ratio'][t])}" for t in TN) + " |")
    e0 = new[both[0]]
    assert all(base[s]["k"] == new[s]["k"] and base[s]["split"] == new[s]["split"] for s in both)
    sar = np.array(e0["sarops"])/1.5
    split = np.array(e0["split"])
    typ = np.array([list(TN)[k % 3] for k in e0["k"]])
    for steps in (both, [s for s in both if s >= 2_000_000]):
        if not steps:
            continue
        pb = np.mean([base[s]["pos"] for s in steps], axis=0)
        pr = np.mean([new[s]["pos"] for s in steps], axis=0)
        d = pr - pb
        lines += ["", f"사례별 짝 비교 (평가 {len(steps)}번의 평균: {', '.join(f'{s/1e6:.1f}M' for s in steps)})", "",
                  "| 묶음 | R10 비율 | R12 비율 | POS 차이 (R12 − R10) 평균 ± 표준오차 | R12가 높은 사례 |", "|---|---|---|---|---|"]
        for name, m in [("전체", np.ones(len(d), bool))] + [(SC[k][0], split == k) for k in SC] + [(TN[t], typ == t) for t in TN]:
            lines.append(f"| {name} | {pb[m].sum()/sar[m].sum():.3f} | {pr[m].sum()/sar[m].sum():.3f} | "
                         f"{d[m].mean():+.4f} ± {d[m].std(ddof=1)/np.sqrt(m.sum()):.4f} | {(d[m] > 0).sum()}/{m.sum()} |")
    lb, lr = (ppo_log(args.runs/p, 0.4) for p in args.logs)
    lb = lb[lb[:, 0] <= last/1e6 + 0.6]
    tbb, tbr = args.runs/args.base/"tb", args.runs/args.run/"tb"
    eb, er = tb_all(tbb, "episode/POS"), tb_all(tbr, "episode/POS")
    lines += ["", "학습 에피소드 POS (최근 100개 평균) 40만 구간 평균, KL 상한에 걸려 멈춘 업데이트 비율", "",
              "| 구간 (만 스텝) | R10 에피소드 POS | R12 에피소드 POS | R10 조기 중단 | R12 조기 중단 |", "|---|---|---|---|---|"]
    for lo in np.arange(0, max(er[0]), 0.4):
        def m(x, y):
            sel = (x >= lo) & (x < lo + 0.4)
            return f"{y[sel].mean():.3f}" if sel.any() else "-"
        def st(L):
            sel = np.abs(L[:, 0] - (lo + 0.2)) < 1e-6
            return f"{L[sel, 1][0]:.0%}" if sel.any() else "-"
        lines.append(f"| {lo*100:.0f}~{(lo + 0.4)*100:.0f} | {m(*eb)} | {m(*er)} | {st(lb)} | {st(lr)} |")
    args.out.mkdir(parents=True, exist_ok=True)
    stem = args.out/"r12_vs_r10"
    stem.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    fig, axes = plt.subplots(2, 3, figsize=(18, 9.5), layout="constrained")
    (a1, a2, a3), (a4, a5, a6) = axes
    xb, xr = np.array(sorted(base))/1e6, np.array(sorted(new))/1e6
    B, R = [base[s] for s in sorted(base)], [new[s] for s in sorted(new)]
    a1.plot(xb, [e["ratio"] for e in B], "--o", color=MUTED, lw=1.8, ms=6, label="R10 두 지도 (미니배치 256, KL 상한 0.02)")
    a1.plot(xr, [e["ratio"] for e in R], "-o", color=C_NEW, lw=2.2, ms=6, label="R12 (미니배치 770, KL 상한 0.03)")
    a1.set(title="검증: SAROPS 대비 비율 (30개)")
    a1.legend(frameon=False, fontsize=9, loc="lower right")
    for k, (name, c) in SC.items():
        a2.plot(xb, [e["by_split"][k] for e in B], "--o", color=c, lw=1.4, ms=4, alpha=.7)
        a2.plot(xr, [e["by_split"][k] for e in R], "-o", color=c, lw=2, ms=5, label=name)
    a2.set(title="지도별 (실선 R12, 점선 R10)")
    a2.legend(frameon=False, fontsize=9, loc="lower right")
    for t, name in TN.items():
        a3.plot(xb, [e["type_ratio"][t] for e in B], "--o", color=TC[t], lw=1.4, ms=4, alpha=.7)
        a3.plot(xr, [e["type_ratio"][t] for e in R], "-o", color=TC[t], lw=2, ms=5, label=name)
    a3.set(title="표적 유형별 (실선 R12, 점선 R10)")
    a3.legend(frameon=False, fontsize=9, loc="lower right")
    for a in (a1, a2, a3):
        a.axhline(1.0, color=MUTED, lw=1, ls="--")
        a.set(xlabel="학습 스텝 (백만)", ylabel="RL / SAROPS (점선 1 = 같음)", ylim=(0.9, 1.4))
    a4.plot(*eb, color=MUTED, lw=1.6, ls="--", label="R10 두 지도")
    a4.plot(*er, color=C_NEW, lw=2, label="R12")
    a4.set(title="학습 에피소드 POS (최근 100 에피소드, 5회 이동 평균)", xlabel="학습 스텝 (백만)", ylabel="에피소드 POS (계획 particle)")
    a4.legend(frameon=False, fontsize=9, loc="lower right")
    for r, (name, c) in {"amboy": SC["nval"], "mojave": SC["bval"]}.items():
        nm = name.split()[0]
        xs, ys = tb_all(tbb, f"episode_region/{r}")
        a5.plot(xs, ys/1.5, color=c, lw=1.4, ls="--", alpha=.7)
        xs, ys = tb_all(tbr, f"episode_region/{r}")
        a5.plot(xs, ys/1.5, color=c, lw=2, label=nm)
    a5.set(title="학습 에피소드 POS, 지도별 (실선 R12, 점선 R10)", xlabel="학습 스텝 (백만)", ylabel="에피소드 POS (계획 particle)")
    a5.legend(frameon=False, fontsize=9, loc="lower right")
    a6.plot(lb[:, 0], lb[:, 1]*100, "--o", color=MUTED, lw=1.8, ms=4, label="R10 두 지도")
    a6.plot(lr[:, 0], lr[:, 1]*100, "-o", color=C_NEW, lw=2.2, ms=5, label="R12")
    a6.set(title="KL 상한에 걸려 중간에 멈춘 PPO 업데이트 (40만 구간)", xlabel="학습 스텝 (백만)", ylabel="%", ylim=(0, 105))
    a6.legend(frameon=False, fontsize=9, loc="center right")
    for a in axes.flat:
        style(a)
        a.set_xlim(0, last/1e6 + 0.5)
    fig.suptitle(f"R12 (처음부터, 미니배치 770·KL 0.03) 대 R10 두 지도: 같은 스텝에서 비교 (R12 검증 {last/1e6:.1f}M까지)", fontsize=12, color=INK)
    fig.savefig(stem.with_suffix(".png"), dpi=95)
    print("saved", stem.with_suffix(".png"))


if __name__ == "__main__":
    main()

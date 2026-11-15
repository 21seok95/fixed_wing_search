"""One in-training evaluation of the two-map R10 run (r10_2map): hold-out POS of RL and SAROPS case by case on nval (Amboy, no-stop)
and bval (Barstow, no-stop, varied routes), and the RL/SAROPS ratio overall, by split and by target type.

    python -m rl_search.scripts.rl_eval.r10_eval_figure --run r10_2map --step 400000 \
        --out rl_search/data/real/amboy/rl_eval/report
Writes <out>/<run>_eval_<step>.png and .md (step defaults to the latest evaluation).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rl_search.scripts.rl_eval.train_report import INK, MUTED, korean_font, style

TYPES = ("road", "mixed", "desert")
TN = {"road": "도로형", "mixed": "혼합형", "desert": "사막형"}
SPLITS = (("nval", "암보이 정지 없음 (nval)"), ("bval", "바스토 경로 다양 (bval)"))
RL_C, SAR_C = "#2a78d6", "#b9b7b0"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="r10_2map")
    ap.add_argument("--runs", type=Path, default=Path("rl_search/data/rl_real"))
    ap.add_argument("--step", type=int, default=None)
    ap.add_argument("--name", default="R10 두 지도", help="run name in the figure title")
    ap.add_argument("--out", type=Path, default=Path("rl_search/data/real/amboy/rl_eval/report"))
    args = ap.parse_args()
    korean_font()
    hist = json.loads((args.runs/args.run/"state.json").read_text())["history"]["3"]
    e = next(h for h in hist if h["step"] == args.step) if args.step else hist[-1]
    rl, sar = np.array(e["pos"]), np.array(e["sarops"])/1.5          # SAROPS totals are 1.5 x POS (a0, w1.5)
    split, k = np.array(e["split"]), np.array(e["k"])
    typ = np.array([TYPES[i % 3] for i in k])

    def ratio(m):
        return rl[m].sum()/sar[m].sum()
    rows = [("전체", np.ones(len(rl), bool))] + [(n.split(" (")[0], split == s) for s, n in SPLITS] + [(TN[t], typ == t) for t in TYPES]
    lines = [f"{args.run}, {e['step']:,} 스텝 평가 (SAROPS 대비 비율 = RL POS 합 / SAROPS POS 합)", "",
             "| 묶음 | 사례 수 | RL POS | SAROPS POS | 비율 | RL이 높은 사례 |", "|---|---|---|---|---|---|"]
    for name, m in rows:
        lines.append(f"| {name} | {m.sum()} | {rl[m].mean():.3f} | {sar[m].mean():.3f} | {ratio(m):.3f} | {(rl[m] > sar[m]).sum()}/{m.sum()} |")
    for s, n in SPLITS:
        for t in TYPES:
            m = (split == s) & (typ == t)
            lines.append(f"| {n.split(' (')[0]} {TN[t]} | {m.sum()} | {rl[m].mean():.3f} | {sar[m].mean():.3f} | {ratio(m):.3f} | {(rl[m] > sar[m]).sum()}/{m.sum()} |")
    args.out.mkdir(parents=True, exist_ok=True)
    stem = args.out/f"{args.run}_eval_{e['step']}"
    stem.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    fig, axes = plt.subplots(1, 3, figsize=(17, 5), layout="constrained", gridspec_kw=dict(width_ratios=[1.6, 1.6, 1]))
    w = 0.4
    for a, (s, n) in zip(axes[:2], SPLITS):
        m = np.flatnonzero(split == s)
        m = m[np.argsort(k[m])]
        x = np.arange(len(m))
        a.bar(x - w/2, rl[m], w*0.92, color=RL_C, label="RL")
        a.bar(x + w/2, sar[m], w*0.92, color=SAR_C, label="SAROPS")
        a.set_xticks(x, [f"{k[i] + 1}\n{TN[typ[i]]}" for i in m], fontsize=8)
        a.set(ylabel="별도 particle POS", ylim=(0, 1.05), title=f"{n}: 비율 {ratio(split == s):.2f}")
        a.legend(frameon=False, fontsize=9, loc="upper right", ncol=2)
    a = axes[2]
    names = ["전체", "암보이\nnval", "바스토\nbval"] + [TN[t] for t in TYPES]
    vals = [ratio(m) for _, m in rows]
    xb = np.arange(len(rows))
    bars = a.bar(xb, vals, 0.6, color=[INK] + [RL_C]*2 + ["#7a52c7"]*3)
    for r, v in zip(bars, vals):
        a.text(r.get_x() + r.get_width()/2, v + 0.01, f"{v:.2f}", ha="center", fontsize=9, color=INK)
    a.axhline(1.0, color=MUTED, lw=1, ls="--")
    a.set_xticks(xb, names, fontsize=9)
    a.set(ylabel="RL / SAROPS (점선 1 = 같음)", title="SAROPS 대비 비율", ylim=(0, max(1.3, max(vals) + 0.1)))
    for a in axes:
        style(a)
    fig.suptitle(f"{args.name} ({args.run}) {e['step']/1e4:,.0f}만 스텝 검증: 사례별 POS (사례 번호 아래는 표적 유형)", fontsize=12, color=INK)
    fig.savefig(stem.with_suffix(".png"), dpi=100)
    print("saved", stem.with_suffix(".png"))


if __name__ == "__main__":
    main()

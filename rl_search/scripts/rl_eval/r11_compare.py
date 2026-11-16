"""R11 (r11_2map: R10 at 4M continued with minibatch 770 and target_kl 0.03) against the plain R10 continuation (r10_2map 4M -> 6M) at
the same total steps: validation ratio to SAROPS overall, by split and by target type, the share of PPO updates cut short by the KL
cap (training logs), and a paired per-case comparison over the evaluations both runs have.

    python -m rl_search.scripts.rl_eval.r11_compare --out rl_search/data/real/amboy/rl_eval/report
Writes <out>/r11_vs_r10.png and .md.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rl_search.scripts.rl_eval.train_report import INK, MUTED, korean_font, style

TN = {"road": "도로형", "mixed": "혼합형", "desert": "사막형"}
TC = {"road": "#2a78d6", "mixed": "#7a52c7", "desert": "#c9772e"}
SC = {"nval": ("암보이 nval", "#2a78d6"), "bval": ("바스토 bval", "#1baf7a")}


def ppo_log(path: Path, bin_m: float = 0.2):
    """Per update (one SB3 table per rollout, train() runs after it): total steps, approx_kl, clip fraction, early stop or not."""
    T, kl, cf, stop = [], [], [], []
    for line in path.read_text(errors="ignore").splitlines():
        if m := re.search(r"total_timesteps\s*\|\s*(\d+)", line):
            T.append(int(m.group(1))); kl.append(np.nan); cf.append(np.nan); stop.append(False)
        elif T and (m := re.search(r"approx_kl\s*\|\s*([\d.e-]+)", line)):
            kl[-1] = float(m.group(1))
        elif T and (m := re.search(r"clip_fraction\s*\|\s*([\d.e-]+)", line)):
            cf[-1] = float(m.group(1))
        elif T and "Early stopping" in line:
            stop[-1] = True
    T, kl, cf, stop = map(np.array, (T, kl, cf, stop))
    out = []
    for lo in np.arange(0, T.max()/1e6, bin_m):
        sel = (T/1e6 >= lo) & (T/1e6 < lo + bin_m)
        if sel.sum() > 5:
            out.append((lo + bin_m/2, stop[sel].mean(), np.nanmean(kl[sel]), np.nanmean(cf[sel]), sel.sum()))
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="r10_2map")
    ap.add_argument("--run", default="r11_2map")
    ap.add_argument("--init-step", type=int, default=4_000_000, help="step of the base run the new run started from")
    ap.add_argument("--runs", type=Path, default=Path("rl_search/data/rl_real"))
    ap.add_argument("--logs", nargs=2, default=["r10_2map_train.log", "r11_train.log"], help="training logs of base and run (in --runs)")
    ap.add_argument("--out", type=Path, default=Path("rl_search/data/real/amboy/rl_eval/report"))
    args = ap.parse_args()
    korean_font()
    hb = json.loads((args.runs/args.base/"state.json").read_text())["history"]["3"]
    hr = json.loads((args.runs/args.run/"state.json").read_text())["history"]["3"]
    s0 = args.init_step
    base = {e["step"]: e for e in hb if e["step"] >= s0}
    new = {s0 + e["step"]: e for e in hr}
    new_x = sorted(new)
    both = [s for s in new_x if s in base]

    lines = [f"{args.run} (R11) 대 {args.base} 이어 학습 (R10), 같은 총 스텝 (R10 {s0/1e6:.0f}M에서 출발)", "",
             "| 총 스텝 (R11 스텝) | R10 전체 | R11 전체 | R10 nval | R11 nval | R10 bval | R11 bval | "
             + " | ".join(f"R10 {TN[t]} | R11 {TN[t]}" for t in TN) + " |", "|" + "---|"*13]
    for s in sorted(set(base) | set(new)):
        b, r = base.get(s), new.get(s)
        f = lambda e, g: f"{g(e):.3f}" if e else "-"
        lines.append(f"| {s/1e6:.1f}M ({(s - s0)/1e6:.1f}M) | {f(b, lambda e: e['ratio'])} | {f(r, lambda e: e['ratio'])} | "
                     + " | ".join(f"{f(b, lambda e, k=k: e['by_split'][k])} | {f(r, lambda e, k=k: e['by_split'][k])}" for k in SC) + " | "
                     + " | ".join(f"{f(b, lambda e, t=t: e['type_ratio'][t])} | {f(r, lambda e, t=t: e['type_ratio'][t])}" for t in TN) + " |")
    if both:
        assert all(base[s]["k"] == new[s]["k"] and base[s]["split"] == new[s]["split"] for s in both)
        pb = np.mean([base[s]["pos"] for s in both], axis=0)
        pr = np.mean([new[s]["pos"] for s in both], axis=0)
        sar = np.array(new[both[0]]["sarops"])/1.5
        d = pr - pb
        lines += ["", f"사례별 짝 비교 (두 run 모두 있는 평가 {len(both)}번의 평균, {', '.join(f'{s/1e6:.1f}M' for s in both)})", "",
                  "| 묶음 | R10 비율 | R11 비율 | POS 차이 (R11 − R10) 평균 ± 표준오차 | R11이 높은 사례 |", "|---|---|---|---|---|"]
        split = np.array(new[both[0]]["split"])
        typ = np.array([list(TN)[k % 3] for k in new[both[0]]["k"]])
        for name, m in [("전체", np.ones(len(d), bool))] + [(SC[k][0], split == k) for k in SC] + [(TN[t], typ == t) for t in TN]:
            lines.append(f"| {name} | {pb[m].sum()/sar[m].sum():.3f} | {pr[m].sum()/sar[m].sum():.3f} | "
                         f"{d[m].mean():+.4f} ± {d[m].std(ddof=1)/np.sqrt(m.sum()):.4f} | {(d[m] > 0).sum()}/{m.sum()} |")
    lb, lr = (ppo_log(args.runs/f) for f in args.logs)
    lr[:, 0] += s0/1e6
    lines += ["", "PPO 업데이트 (20만 스텝 구간 평균): KL 상한에 걸려 중간에 멈춘 업데이트 비율, approx_kl, clip 비율", "",
              "| 구간 중심 (총 스텝) | run | 조기 중단 비율 | approx_kl | clip 비율 | 업데이트 수 |", "|---|---|---|---|---|---|"]
    for name, L in (("R10", lb[lb[:, 0] > s0/1e6 - 0.4]), ("R11", lr)):
        for x, st, kl, cf, n in L:
            lines.append(f"| {x:.1f}M | {name} | {st:.0%} | {kl:.4f} | {cf:.3f} | {n:.0f} |")
    args.out.mkdir(parents=True, exist_ok=True)
    stem = args.out/"r11_vs_r10"
    stem.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    fig, (a1, a2, a3, a4) = plt.subplots(1, 4, figsize=(21, 5), layout="constrained")
    xb = np.array(sorted(base))/1e6
    xr = np.r_[s0, new_x]/1e6                                     # R11 starts from the R10 point at s0
    start = base[s0]
    er = [start] + [new[s] for s in new_x]
    a1.plot(xb, [base[s]["ratio"] for s in sorted(base)], "--o", color=MUTED, lw=1.8, ms=6, label="R10 그대로 이어 학습")
    a1.plot(xr, [e["ratio"] for e in er], "-o", color="#2a78d6", lw=2.2, ms=6, label="R11 (미니배치 770, KL 상한 0.03)")
    a1.set(title="검증: SAROPS 대비 비율 (30개)", ylabel="RL / SAROPS")
    a1.legend(frameon=False, fontsize=9, loc="lower right")
    for k, (name, c) in SC.items():
        a2.plot(xb, [base[s]["by_split"][k] for s in sorted(base)], "--o", color=c, lw=1.4, ms=4, alpha=.7)
        a2.plot(xr, [e["by_split"][k] for e in er], "-o", color=c, lw=2, ms=5, label=f"{name} (실선 R11, 점선 R10)")
    a2.set(title="지도별 (실선 R11, 점선 R10)", ylabel="RL / SAROPS")
    a2.legend(frameon=False, fontsize=9, loc="lower right")
    for t, name in TN.items():
        a3.plot(xb, [base[s]["type_ratio"][t] for s in sorted(base)], "--o", color=TC[t], lw=1.4, ms=4, alpha=.7)
        a3.plot(xr, [e["type_ratio"][t] for e in er], "-o", color=TC[t], lw=2, ms=5, label=name)
    a3.set(title="표적 유형별 (실선 R11, 점선 R10)", ylabel="RL / SAROPS")
    a3.legend(frameon=False, fontsize=9, loc="lower right")
    for a in (a1, a2, a3):
        a.set(xlabel="총 학습 스텝 (백만, R11은 R10 400만에서 출발)", ylim=(1.05, 1.4))
        a.axvline(s0/1e6, color=MUTED, lw=.8, ls=":")
    a4.plot(lb[:, 0], lb[:, 1]*100, "--o", color=MUTED, lw=1.8, ms=4, label="R10 (미니배치 256, KL 상한 0.02)")
    a4.plot(lr[:, 0], lr[:, 1]*100, "-o", color="#2a78d6", lw=2.2, ms=5, label="R11 (미니배치 770, KL 상한 0.03)")
    a4.set(xlabel="총 학습 스텝 (백만)", ylabel="KL 상한에 걸려 멈춘 업데이트 (%)", title="PPO 업데이트가 중간에 멈춘 비율", ylim=(0, 105))
    a4.legend(frameon=False, fontsize=9, loc="center left")
    for a in (a1, a2, a3, a4):
        style(a)
    fig.suptitle(f"R11 대 R10 이어 학습: 같은 총 스텝에서 비교 (R11 {new_x[-1] - s0:,} 스텝까지)", fontsize=12, color=INK)
    fig.savefig(stem.with_suffix(".png"), dpi=100)
    print("saved", stem.with_suffix(".png"))


if __name__ == "__main__":
    main()

"""Progress of the two-map R10 run (r10_2map): validation ratio to SAROPS (all, nval, bval) at every in-training evaluation, with the R9
curve on its own validation set (mval, stopping targets) for reference, and the training episodes by map and by target type
(TensorBoard, last 100 episodes; hold-out POS = total / 1.5).

    python -m rl_search.scripts.rl_eval.r10_curve --run r10_2map --out rl_search/data/real/amboy/rl_eval/report
Writes <out>/<run>_curve.png and .md.
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

TN = {"road": "도로형", "mixed": "혼합형", "desert": "사막형"}
TC = {"road": "#2a78d6", "mixed": "#7a52c7", "desert": "#c9772e"}
RC = {"amboy": ("암보이", "#2a78d6"), "mojave": ("바스토", "#1baf7a")}


def tb_series(tb: Path, tag: str, k: int = 5):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    ea = EventAccumulator(str(sorted(tb.glob("ppo_*"))[-1]), size_guidance={"scalars": 0})
    ea.Reload()
    if tag not in ea.Tags()["scalars"]:
        return np.array([]), np.array([])
    s = ea.Scalars(tag)
    x, y = np.array([e.step for e in s])/1e6, np.array([e.value for e in s])
    if len(y) > k:                                   # light running mean over k rollouts (the buffer itself spans ~100 episodes)
        y = np.convolve(y, np.ones(k)/k, mode="valid")
        x = x[k - 1:]
    return x, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="r10_2map")
    ap.add_argument("--ref", default="amboy_r9")
    ap.add_argument("--runs", type=Path, default=Path("rl_search/data/rl_real"))
    ap.add_argument("--out", type=Path, default=Path("rl_search/data/real/amboy/rl_eval/report"))
    args = ap.parse_args()
    korean_font()
    hist = json.loads((args.runs/args.run/"state.json").read_text())["history"]["3"]
    ref = json.loads((args.runs/args.ref/"state.json").read_text())["history"]["3"] if (args.runs/args.ref/"state.json").exists() else []
    tb = args.runs/args.run/"tb"

    lines = [f"{args.run} 검증 (SAROPS 대비 비율)", "", "| 스텝 | 전체 | 암보이 nval | 바스토 bval | 도로형 | 혼합형 | 사막형 |", "|---|---|---|---|---|---|---|"]
    for e in hist:
        lines.append(f"| {e['step']:,} | {e['ratio']:.3f} | {e['by_split']['nval']:.3f} | {e['by_split']['bval']:.3f} | "
                     + " | ".join(f"{e['type_ratio'][t]:.3f}" for t in TN) + " |")
    lines += ["", "학습 에피소드 POS (최근 100개 평균, 구간 평균)", "", "| 구간 (만 스텝) | 전체 | 암보이 | 바스토 | 도로형 | 혼합형 | 사막형 |",
              "|---|---|---|---|---|---|---|"]
    raw = {t: tb_series(tb, t, 1) for t in ["episode/POS"] + [f"episode_region/{r}" for r in RC] + [f"episode_type/{t}" for t in TN]}
    last = raw["episode/POS"][0].max()
    for lo in np.arange(0, last, 0.2):
        def m(t, div):
            x, y = raw[t]
            sel = (x >= lo) & (x < lo + 0.2)
            return f"{y[sel].mean()/div:.3f}" if sel.any() else "-"
        lines.append(f"| {lo*100:.0f}~{(lo + 0.2)*100:.0f} | {m('episode/POS', 1)} | " + " | ".join(m(f"episode_region/{r}", 1.5) for r in RC)
                     + " | " + " | ".join(m(f"episode_type/{t}", 1.5) for t in TN) + " |")
    args.out.mkdir(parents=True, exist_ok=True)
    stem = args.out/f"{args.run}_curve"
    stem.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(17, 5), layout="constrained")
    x = np.array([e["step"] for e in hist])/1e6
    a1.plot(x, [e["ratio"] for e in hist], "-o", color=INK, lw=2, ms=6, label="R10 두 지도: 전체 30개")
    a1.plot(x, [e["by_split"]["nval"] for e in hist], "-o", color=RC["amboy"][1], lw=1.5, ms=5, label="  암보이 nval 15개")
    a1.plot(x, [e["by_split"]["bval"] for e in hist], "-o", color=RC["mojave"][1], lw=1.5, ms=5, label="  바스토 bval 15개")
    if ref:
        a1.plot([e["step"]/1e6 for e in ref], [e["ratio"] for e in ref], ":", color=MUTED, lw=1.5, label="참고: R9 (다른 검증 세트 mval)")
    a1.axhline(1.0, color=MUTED, lw=1, ls="--")
    a1.set(xlabel="학습 스텝 (백만)", ylabel="RL / SAROPS (점선 1 = 같음)", title="검증: SAROPS 대비 비율", xlim=(0, 4.1), ylim=(0.8, 1.45))
    a1.legend(frameon=False, fontsize=9, loc="upper left")
    for r, (name, c) in RC.items():
        xs, ys = tb_series(tb, f"episode_region/{r}")
        a2.plot(xs, ys/1.5, color=c, lw=1.6, label=name)
    xs, ys = tb_series(tb, "episode/POS")
    a2.plot(xs, ys, color=INK, lw=2, label="전체")
    a2.set(xlabel="학습 스텝 (백만)", ylabel="에피소드 POS (계획 particle)", title="학습 에피소드: 지도별", ylim=(0, None))
    a2.legend(frameon=False, fontsize=9, loc="lower right")
    for t, name in TN.items():
        xs, ys = tb_series(tb, f"episode_type/{t}")
        a3.plot(xs, ys/1.5, color=TC[t], lw=1.6, label=name)
    a3.set(xlabel="학습 스텝 (백만)", ylabel="에피소드 POS (계획 particle)", title="학습 에피소드: 표적 유형별", ylim=(0, None))
    a3.legend(frameon=False, fontsize=9, loc="lower right")
    for a in (a1, a2, a3):
        style(a)
    fig.suptitle(f"R10 두 지도 ({args.run}) 진행: {last:.2f}M 스텝 (학습 곡선은 최근 100 에피소드 평균, 5회 이동 평균)", fontsize=12, color=INK)
    fig.savefig(stem.with_suffix(".png"), dpi=100)
    print("saved", stem.with_suffix(".png"))


if __name__ == "__main__":
    main()

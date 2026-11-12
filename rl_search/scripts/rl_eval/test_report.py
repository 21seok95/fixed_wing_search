"""RL against SAROPS on the test cases (docs/rl_mdp_design.md, section 1: the main comparison, scored on hold-out particles):
paired differences, a scatter of the cases, the POS and A parts, and the paths of the largest RL win, the median and the largest SAROPS
win.

    python -m rl_search.scripts.rl_eval.test_report --tag amboy_v1 --label 1000000
Needs <out>/<tag>/eval_test_<label>.json (eval_rl.py) and rl_eval/sarops_ref_test.json (sarops_ref.py --split test).
Writes <out>/<tag>/test_<label>_compare.png, test_<label>_paths.png and test_<label>.md.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rl_search.scripts.rl_eval.train_report import C_RL, C_SAR, INK, MUTED, decode, korean_font, load_run, style


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", type=Path, default=Path("rl_search/data/rl_real"))
    args = ap.parse_args()
    from rl_search.rl_real_eval import run_area_def, sarops_refs
    from rl_search.scripts.real_map import vehicle_web  # noqa: F401  (resets the font, so import before drawing)
    korean_font()
    d = args.out/args.tag
    lab = f"{int(args.label):,}" if args.label.isdigit() else args.label
    rows = sorted(json.loads((d/f"eval_test_{args.label}.json").read_text()), key=lambda r: (r["stage"], r["k"]))
    refs = sarops_refs("test", run_area_def(args.out/args.tag))
    rows = [r for r in rows if (r["stage"], r["k"]) in refs]
    rl = np.array([r["holdout"]["total"] for r in rows])
    sa = np.array([refs[(r["stage"], r["k"])]["holdout"]["total"] for r in rows])
    rl_p, rl_a = np.array([r["holdout"]["pos"] for r in rows]), np.array([r["holdout"]["A"] for r in rows])
    sa_p = np.array([refs[(r["stage"], r["k"])]["holdout"]["pos"] for r in rows])
    sa_a = np.array([refs[(r["stage"], r["k"])]["holdout"]["A"] for r in rows])
    diff = rl - sa
    n = len(rows)
    se = lambda x: x.std(ddof=1)/np.sqrt(len(x))
    lines = [f"# RL {lab} 스텝 vs SAROPS, 시험 {n}개 (3단계 이륙, 별도 particle)", "",
             "| 지표 | RL | SAROPS | 차이 (RL − SAROPS) ± 표준오차 | RL이 높은 경우 |", "|---|---|---|---|---|"]
    for name, a, b in (("A + POS", rl, sa), ("POS", rl_p, sa_p), ("A", rl_a, sa_a)):
        dd = a - b
        lines.append(f"| {name} | {a.mean():.3f} | {b.mean():.3f} | {dd.mean():+.3f} ± {se(dd):.3f} | {(dd > 0).sum()}/{n} |")
    lines += ["", "| 경우 | RL A + POS | SAROPS A + POS | 차이 |", "|---|---|---|---|"]
    lines += [f"| {r['k'] + 1} | {rl[i]:.3f} | {sa[i]:.3f} | {diff[i]:+.3f} |" for i, r in enumerate(rows)]
    (d/f"test_{args.label}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:7]))

    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(17, 5.2), layout="constrained")
    lo, hi = min(rl.min(), sa.min())*0.95, max(rl.max(), sa.max())*1.04
    a1.plot([lo, hi], [lo, hi], color=MUTED, lw=1)
    a1.scatter(sa, rl, s=55, color=C_RL, edgecolor="white", lw=1.2, zorder=3)
    a1.text(.04, .94, f"대각선 위 = RL이 높음: {(diff > 0).sum()}/{n}", transform=a1.transAxes, color=MUTED, fontsize=10, va="top")
    a1.set(xlim=(lo, hi), ylim=(lo, hi), aspect="equal", xlabel="SAROPS A + POS", ylabel="RL A + POS", title="시험 경우별 점수")
    order = np.argsort(diff)
    a2.bar(np.arange(n), diff[order], color=[C_RL if v > 0 else C_SAR for v in diff[order]], width=.75)
    a2.axhline(0, color=MUTED, lw=1)
    a2.axhline(diff.mean(), color=INK, lw=1.2, ls="--")
    a2.text(0, diff.mean(), f" 평균 {diff.mean():+.3f} ± {se(diff):.3f}", va="bottom", fontsize=10, color=INK)
    a2.set(xticks=[], xlabel="경우 (차이 순서)", ylabel="RL − SAROPS (A + POS)", title="짝지은 차이")
    x = np.arange(2)
    w = .35
    a3.bar(x - w/2, [sa_p.mean(), rl_p.mean()], w, color=[C_SAR, C_RL], label="POS")
    a3.bar(x + w/2, [sa_a.mean(), rl_a.mean()], w, color=["#9cc3ee", "#f6b89a"], label="A")
    for i, (p, a) in enumerate(((sa_p, sa_a), (rl_p, rl_a))):
        a3.text(i - w/2, p.mean() + .01, f"{p.mean():.3f}", ha="center", fontsize=10)
        a3.text(i + w/2, a.mean() + .01, f"{a.mean():.3f}", ha="center", fontsize=10)
    a3.set(xticks=x, xticklabels=["SAROPS", f"RL {lab}"], ylabel="평균", title="두 항: 왼쪽 POS, 오른쪽 A")
    for a in (a1, a2, a3):
        style(a)
    fig.suptitle(f"시험 {n}개 (LKP에서 8~20 km 고리에서 이륙, 별도 particle로 채점): RL {rl.mean():.3f} vs SAROPS {sa.mean():.3f}", fontsize=13)
    fig.savefig(d/f"test_{args.label}_compare.png", dpi=100)
    plt.close(fig)

    hi_i, lo_i = int(order[-1]), int(order[0])                 # titles follow the sign: one side may win every case
    picks = [("RL이 가장 많이 앞선 경우" if diff[hi_i] > 0 else "SAROPS가 가장 적게 앞선 경우", hi_i), ("중간", int(order[n//2])),
             ("SAROPS가 가장 많이 앞선 경우" if diff[lo_i] < 0 else "RL이 가장 적게 앞선 경우", lo_i)]
    fig, axs = plt.subplots(2, 3, figsize=(14, 9), layout="constrained")
    for col, (title, i) in enumerate(picks):
        r = rows[i]
        for row, (rid, color, name) in enumerate(((f"eval-test-s{r['stage']}-k{r['k']}-sarops", C_SAR, "SAROPS"),
                                                  (f"eval-test-s{r['stage']}-k{r['k']}-rl-{args.label}", C_RL, f"RL {lab}"))):
            a = axs[row, col]
            run = load_run(rid)
            if run is None:
                a.text(.5, .5, f"{rid} 없음", ha="center", transform=a.transAxes)
                continue
            o, rs, m = run["opt"], run["res"], run["meta"]
            T, N = int(rs["T"]), int(rs["N"])
            P = decode(rs["pos"], "<i2", (T, N, 2)).astype(float)*float(rs["pos_unit_m"])
            W = decode(o["weight"], np.uint8, (int(o["T"]), int(o["N"])))
            xy = decode(o["uavs"][0]["xy"], np.float32, (-1, 2))
            j = min(int(o["T"]) - 1, T - 1)
            live = W[j] > 0
            a.scatter(P[j, live, 0]/1000, P[j, live, 1]/1000, c=W[j, live]/255, s=1.5, cmap="Greys", vmin=0, vmax=1, zorder=1)
            a.plot(xy[:, 0]/1000, xy[:, 1]/1000, color=color, lw=1.1, zorder=2)
            a.plot(*np.array(o["start"])/1000, marker="^", color=INK, ms=7, zorder=3)
            a.plot(*np.array(o["lkp"])/1000, marker="*", color="#f2c94c", mec=INK, ms=12, zorder=3)
            h = m["holdout"]
            a.set_title(f"{title} (경우 {r['k'] + 1})\n{name}: A + POS {h['total']:.3f} (POS {h['pos']:.3f}, A {h['A']:.3f})", fontsize=10)
            a.set_aspect("equal")
            a.tick_params(labelsize=8)
    fig.suptitle("시험 경로: 위 SAROPS, 아래 RL. 점 = 비행 끝 시각의 particle (진할수록 남은 가중치 큼), ★ LKP, ▲ 이륙점", fontsize=12)
    fig.savefig(d/f"test_{args.label}_paths.png", dpi=95)
    plt.close(fig)
    print("saved", d/f"test_{args.label}_compare.png", d/f"test_{args.label}_paths.png")


if __name__ == "__main__":
    main()

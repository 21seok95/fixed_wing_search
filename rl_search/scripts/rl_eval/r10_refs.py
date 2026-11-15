"""SAROPS references of R10 (docs/r10_ubuntu.md, step 3): hold-out POS of SAROPS case by case on the validation cases with stopping
targets (mval, from the R8 run's state.json: the same reference key with stops) and without (nval), and the Barstow cases (bval) when
they are made; means by target type. SAROPS plans and scores on POS only (a0), so POS = total / 1.5.

    python -m rl_search.scripts.rl_eval.r10_refs --out rl_search/data/real/amboy/rl_eval/report
Writes <out>/r10_sarops_refs.png and .md.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rl_search.scripts.rl_eval.train_report import INK, korean_font, style

KEY = "realtime_t5_d3z5l15_s75_o150_i0.75_a0_w1.5"
TYPES = ("road", "mixed", "desert")
TN = {"road": "도로형", "mixed": "혼합형", "desert": "사막형"}
SETS = (("mval", "암보이, 정지 있음 (mval)", "#9cc3ee"), ("nval", "암보이, 정지 없음 (nval)", "#2a78d6"), ("bval", "바스토, 정지 없음·경로 다양 (bval)", "#1baf7a"))


def pos_by_case(split: str, root: Path) -> dict:
    """Hold-out POS by case k: the reference file, or for mval the SAROPS totals the R8 run kept (same key, mval made on the Mac)."""
    if split == "mval":
        h = json.loads((root/"amboy_r8"/"state.json").read_text())["history"]["3"][0]
        return {k: v/1.5 for k, v in enumerate(h["sarops"])}
    from rl_search.scripts.rl_eval.sarops_ref import ref_path
    p = ref_path(split, KEY)
    if not p.exists():
        return {}
    return {r["case"]["k"]: float(r["holdout"]["pos"]) for r in json.loads(p.read_text()) if r["case"]["stage"] == 3}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=Path("rl_search/data/rl_real"))
    ap.add_argument("--out", type=Path, default=Path("rl_search/data/real/amboy/rl_eval/report"))
    args = ap.parse_args()
    korean_font()
    data = {sp: pos_by_case(sp, args.runs) for sp, _, _ in SETS}
    sets = [s for s in SETS if data[s[0]]]
    lines = ["| 묶음 | 사례 수 | 전체 | " + " | ".join(TN[t] for t in TYPES) + " |", "|---|---|---|---|---|---|"]
    for sp, name, _ in sets:
        d = data[sp]
        by = [np.mean([v for k, v in d.items() if k % 3 == i]) for i in range(3)]
        lines.append(f"| {name} | {len(d)} | {np.mean(list(d.values())):.3f} | " + " | ".join(f"{v:.3f}" for v in by) + " |")
    ks = sorted(set(data["mval"]) & set(data["nval"]))
    if ks:
        dd = np.array([data["nval"][k] - data["mval"][k] for k in ks])
        lines += ["", f"정지 없음 − 정지 있음 (암보이 같은 사례 {len(ks)}개): {dd.mean():+.3f} ± {dd.std(ddof=1)/np.sqrt(len(dd)):.3f}, "
                      f"정지 없음이 높은 사례 {(dd > 0).sum()}/{len(dd)}"]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/"r10_sarops_refs.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 5), layout="constrained", gridspec_kw=dict(width_ratios=[2.2, 1]))
    n = max(len(data[s[0]]) for s in sets)
    x = np.arange(n)
    w = 0.8/len(sets)
    for j, (sp, name, c) in enumerate(sets):
        d = data[sp]
        a1.bar(x + (j - (len(sets) - 1)/2)*w, [d.get(k, np.nan) for k in range(n)], w*0.92, color=c, label=name)
    a1.set_xticks(x, [f"{k + 1}\n{TN[TYPES[k % 3]]}" for k in range(n)], fontsize=8)
    a1.set(ylabel="SAROPS 별도 particle POS", title="사례별 SAROPS POS (사례 번호가 같으면 LKP·이륙 시각·출발점이 같음, 바스토는 다른 사례)")
    a1.legend(frameon=False, fontsize=9, loc="upper left", ncol=len(sets))
    a1.set_ylim(0, 1.05)
    xt = np.arange(len(TYPES) + 1)
    for j, (sp, name, c) in enumerate(sets):
        d = data[sp]
        vals = [np.mean(list(d.values()))] + [np.mean([v for k, v in d.items() if k % 3 == i]) for i in range(3)]
        b = a2.bar(xt + (j - (len(sets) - 1)/2)*w, vals, w*0.92, color=c)
        for r, v in zip(b, vals):
            a2.text(r.get_x() + r.get_width()/2, v + 0.01, f"{v:.2f}", ha="center", fontsize=8, color=INK)
    a2.set_xticks(xt, ["전체"] + [TN[t] for t in TYPES])
    a2.set(ylabel="평균 POS", title="유형별 평균", ylim=(0, 1))
    for a in (a1, a2):
        style(a)
    fig.suptitle("R10 비교 기준: SAROPS (POS만으로 계획, 별도 particle로 채점)", fontsize=12, color=INK)
    fig.savefig(args.out/"r10_sarops_refs.png", dpi=100)
    print("saved", args.out/"r10_sarops_refs.png", "| sets:", ", ".join(s[0] for s in sets))


if __name__ == "__main__":
    main()

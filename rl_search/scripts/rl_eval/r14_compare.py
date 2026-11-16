"""R14 type specialists (r14_road: road cases only, r14_desert: desert cases only; R13's settings) against models that learn all three types,
on the validation cases of that type at the same steps: ratio to SAROPS (RL POS sum / SAROPS POS sum over the 10 cases, nval 5 + bval 5),
the training episodes of that type (TensorBoard episode_type/<type>, hold-out POS = total / 1.5), and circling by flight phase on those
cases (circle_check.py JSON, optional). R13 (all three types, the same settings) is the fair reference; R10 and R12 (four maps, A' 0.01)
are shown for reference when R13 is not here.

    python -m rl_search.scripts.rl_eval.r14_compare --refs r13_2map r10_2map r12_2map \
        --circle "R14 도로형 120만=report/circle_r14_road-1200000.json" "R12 120만=report/circle_r12_2map-1200000.json"
Writes <out>/r14_compare.png and .md.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rl_search.scripts.rl_eval.r12_compare import tb_all
from rl_search.scripts.rl_eval.train_report import INK, MUTED, korean_font, style

TN = {"road": "도로형", "mixed": "혼합형", "desert": "사막형"}
TYPES = ("road", "mixed", "desert")
REF_STYLE = {"r13_2map": ("R13 (세 유형, 같은 설정)", INK, "-"), "r10_2map": ("R10 (세 유형, 지도 4장·A′ 0.01)", MUTED, "--"),
             "r12_2map": ("R12 (세 유형, 지도 4장·A′ 0.01)", "#b9b7b0", ":")}
C_SPEC = {"road": "#2a78d6", "desert": "#c9772e"}
PHASES = (0, 15, 30, 45)


def type_curve(hist, typ):
    """Ratio to SAROPS on the cases of one type at every evaluation (case k has type k % 3)."""
    xs, ys = [], []
    for e in hist:
        idx = [i for i, k in enumerate(e["k"]) if TYPES[k % 3] == typ]
        if idx:
            sar = np.array(e["sarops"])[idx]/1.5
            xs.append(e["step"]/1e6)
            ys.append(float(np.array(e["pos"])[idx].sum()/sar.sum()))
    return np.array(xs), np.array(ys)


def cases(hist, step, typ):
    """{(split, k): (RL POS, SAROPS POS)} on the cases of one type at one evaluation, or None if the run has no evaluation at that step."""
    e = next((e for e in hist if e["step"] == step), None)
    return e and {(sp, k): (p, s/1.5) for sp, k, p, s in zip(e["split"], e["k"], e["pos"], e["sarops"]) if TYPES[k % 3] == typ}


def circling(path: Path, typ: str):
    rows = [r for r in json.loads(path.read_text()) if r["typ"] == typ]
    out = []
    for lo in PHASES:
        rr = [r for r in rows if lo <= r["m"] < lo + 15]
        c = np.array([r["net_turn"] >= 330 for r in rr], bool)
        g = np.array([r["gain"] for r in rr])
        out.append((c.mean() if len(c) else np.nan, g[c].mean() if c.any() else np.nan, g[~c].mean() if (~c).any() else np.nan, len(rr)))
    return out, len({(r["sp"], r["k"]) for r in rows})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--specs", nargs="+", default=["r14_road", "r14_desert"])
    ap.add_argument("--refs", nargs="+", default=["r13_2map", "r10_2map", "r12_2map"], help="all-type runs (skipped when not here)")
    ap.add_argument("--final", nargs="*", default=["R10 600만=r10_2map:6000000", "R12 600만=r12_2map:6000000",
                                                    "R11 총 600만 (지금까지 최고)=r11_2map:2000000"],
                    help='"name=run:step" of finished models for the paired table (R11 counts its steps from R10 4M)')
    ap.add_argument("--circle", nargs="*", default=[], help='"name=circle json" (circle_check.py) for the circling panel')
    ap.add_argument("--runs", type=Path, default=Path("rl_search/data/rl_real"))
    ap.add_argument("--out", type=Path, default=Path("rl_search/data/real/amboy/rl_eval/report"))
    args = ap.parse_args()
    korean_font()
    specs = [(r, r.split("_")[-1]) for r in args.specs if (args.runs/r/"state.json").exists()]
    refs = [r for r in args.refs if (args.runs/r/"state.json").exists()]
    H = {r: json.loads((args.runs/r/"state.json").read_text())["history"]["3"] for r in [s for s, _ in specs] + refs}
    circ = [(n, Path(p)) for n, p in (c.split("=", 1) for c in args.circle)]
    finals = [(n, *rs.rsplit(":", 1)) for n, rs in (f.split("=", 1) for f in args.final)]
    for _, r, _ in finals:
        if r not in H and (args.runs/r/"state.json").exists():
            H[r] = json.loads((args.runs/r/"state.json").read_text())["history"]["3"]

    lines = ["R14 유형 전문 모델 대 세 유형 모델: 그 유형 검증 사례 10개(nval 5 + bval 5)에서 SAROPS 대비 비율, 같은 스텝", ""]
    for run, typ in specs:
        cols = [run] + refs
        curves = {r: dict(zip(*[np.round(v, 2) if i == 0 else v for i, v in enumerate(type_curve(H[r], typ))])) for r in cols}
        steps = sorted(set().union(*[c.keys() for c in curves.values()]))
        steps = [s for s in steps if s <= max(curves[run].keys(), default=0) + 1e-9]
        lines += [f"### {TN[typ]} ({run})", "", "| 스텝 (백만) | " + " | ".join(cols) + " |", "|---|" + "---|"*len(cols)]
        for s in steps:
            lines.append(f"| {s:.1f} | " + " | ".join(f"{curves[r][s]:.3f}" if s in curves[r] else "-" for r in cols) + " |")
        lines.append("")
        last = H[run][-1]["step"]
        a = cases(H[run], last, typ)
        keys = sorted(a)
        sar, pa = (np.array([a[k][i] for k in keys]) for i in (1, 0))
        lines += [f"{run} {last/1e6:.1f}M과 사례별 짝 비교 ({TN[typ]} 사례 {len(keys)}개, R14 비율 {pa.sum()/sar.sum():.3f})", "",
                  "| 비교 모델 | 그 모델 비율 | POS 차이 (R14 − 비교) 평균 ± 표준오차 | R14가 높은 사례 |", "|---|---|---|---|"]
        for name, r, st in [(f"{r} {last/1e6:.1f}M (같은 스텝)", r, last) for r in refs] + finals:
            b = cases(H.get(r, []), int(st), typ)
            if b:
                d = pa - np.array([b[k][0] for k in keys])
                lines.append(f"| {name} | {sum(b[k][0] for k in keys)/sar.sum():.3f} | {d.mean():+.4f} ± {d.std(ddof=1)/np.sqrt(len(d)):.4f} | "
                             f"{(d > 0).sum()}/{len(d)} |")
        lines.append("")
        if circ:
            lines += [f"{TN[typ]} 사례의 원 선회 (앞뒤 3분 동안 한 방향 330° 이상인 분의 몫, 괄호: 분당 POS 원 선회 / 나머지)", "",
                      "| 비행 | 비행 수 | " + " | ".join(f"{lo}~{lo + 15}분" for lo in PHASES) + " |", "|---|---|" + "---|"*len(PHASES)]
            for n, p in circ:
                ph, nf = circling(p, typ)
                if nf:
                    lines.append(f"| {n} | {nf} | " + " | ".join(f"{c*100:.0f}% ({a:.4f} / {b:.4f})" for c, a, b, _ in ph) + " |")
            lines.append("")
    args.out.mkdir(parents=True, exist_ok=True)
    stem = args.out/"r14_compare"
    stem.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    ncol = 3 if circ else 2
    fig, axes = plt.subplots(len(specs), ncol, figsize=(6*ncol, 4.8*len(specs)), layout="constrained", squeeze=False)
    for row, (run, typ) in zip(axes, specs):
        a = row[0]
        x, y = type_curve(H[run], typ)
        a.plot(x, y, "-o", color=C_SPEC[typ], lw=2.4, ms=6, label=f"R14 {TN[typ]}만 ({run})")
        for r in refs:
            xr, yr = type_curve(H[r], typ)
            sel = xr <= x.max() + 0.41
            name, c, ls = REF_STYLE.get(r, (r, MUTED, "--"))
            a.plot(xr[sel], yr[sel], ls, marker="o", color=c, lw=1.6, ms=4, label=name)
        a.axhline(1.0, color=MUTED, lw=1, ls="--")
        a.set(xlabel="학습 스텝 (백만)", ylabel="RL / SAROPS (점선 1 = 같음)", title=f"검증: {TN[typ]} 사례 10개")
        a.legend(frameon=False, fontsize=8.5, loc="lower right")
        a = row[1]
        xs, ys = tb_all(args.runs/run/"tb", f"episode_type/{typ}")
        if len(xs):
            a.plot(xs, ys/1.5, color=C_SPEC[typ], lw=2, label=f"R14 {TN[typ]}만")
        for r in refs:
            xr, yr = tb_all(args.runs/r/"tb", f"episode_type/{typ}")
            sel = xr <= (xs.max() if len(xs) else 0) + 0.2
            name, c, ls = REF_STYLE.get(r, (r, MUTED, "--"))
            a.plot(xr[sel], yr[sel]/1.5, ls, color=c, lw=1.4, label=name)
        a.set(xlabel="학습 스텝 (백만)", ylabel="에피소드 POS (계획 particle)", title=f"학습 에피소드: {TN[typ]} (최근 100개, 5회 이동 평균)", ylim=(0, None))
        a.legend(frameon=False, fontsize=8.5, loc="lower right")
        if circ:
            a = row[2]
            have = [(n, *circling(p, typ)) for n, p in circ]
            have = [h for h in have if h[2]]                          # flights of this type only (a specialist's file has one type)
            wb = 0.8/len(have)
            for i, (n, ph, nf) in enumerate(have):
                c = next((v[1] for r, v in REF_STYLE.items() if n.startswith(r[:3].upper())), C_SPEC[typ])
                a.bar(np.arange(len(PHASES)) + (i - (len(have) - 1)/2)*wb, [v[0]*100 for v in ph], wb*0.92, color=c, label=f"{n} ({nf}개)")
            a.set_xticks(range(len(PHASES)), [f"{lo}~{lo + 15}분" for lo in PHASES])
            a.set(ylabel="원 선회한 분의 몫 (%)", title=f"{TN[typ]} 사례의 원 선회 (이륙 뒤 시간대)", ylim=(0, a.get_ylim()[1]*1.3))   # room for the legend
            a.legend(frameon=False, fontsize=8.5, loc="upper left")
        for a in row:
            style(a)
    fig.suptitle("R14 유형 전문 모델 대 세 유형을 함께 학습한 모델 (R13 설정; R10·R12는 지도 4장·A′ 0.01이라 참고용)", fontsize=12, color=INK)
    fig.savefig(stem.with_suffix(".png"), dpi=95)
    print("saved", stem.with_suffix(".png"))


if __name__ == "__main__":
    main()

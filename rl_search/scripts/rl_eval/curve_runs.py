"""Validation curves of training runs: ratio of the hold-out score to SAROPS, all cases and by scenario type, at every in-training
evaluation (<run>/state.json) and at checkpoints scored afterwards with eval_rl (web runs eval-<split>-s3-k<k>-rl-<run>-<step>).
Optional reference lines: a flight set scored on the same split (e.g. the R9 model on the no-stop cases).

    python -m rl_search.scripts.rl_eval.curve_runs --runs amboy_r10 --ref "R9 400만=amboy_r9-4000000" --split nval \
        --out rl_search/data/real/amboy/rl_eval/report/r10_curve.png
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np

TYPES = [("all", "전체"), ("road", "도로형"), ("mixed", "혼합형"), ("desert", "사막형")]
COLORS = ["#d6336c", "#1f1f1e", "#2a6db0", "#7a52c7", "#c9772e"]


def sarops_of(split: str, key: str) -> dict:
    from rl_search.rl_real_eval import sarops_refs
    return {k: r["holdout"]["total"] for (st, k), r in sarops_refs(split, key).items() if st == 3}


def flight_set(split: str, label: str) -> tuple[dict, str]:
    """Hold-out totals by case of the web runs eval-<split>-s3-k<k>-rl-<label>, and their reference key."""
    from rl_search.scripts.real_map.vehicle_web import RUNS
    out, key = {}, None
    for d in RUNS.glob(f"eval-{split}-s3-k*-rl-{label}"):
        m = json.loads((d/"meta.json").read_text())
        out[int(m["case"])] = float(m["holdout"]["total"])
        key = m["holdout"].get("ref_key")
    return out, key


def ratios(rl: dict, sar: dict) -> dict:
    ks = sorted(set(rl) & set(sar))
    r, s = np.array([rl[k] for k in ks]), np.array([sar[k] for k in ks])
    t = np.array(ks) % 3
    out = {"all": r.mean()/s.mean()}
    for i, n in enumerate(["road", "mixed", "desert"]):
        if (t == i).any():
            out[n] = r[t == i].mean()/s[t == i].mean()
    return out


def run_points(run: str, split: str, root: Path) -> tuple[np.ndarray, dict]:
    hist = json.loads((root/run/"state.json").read_text())["history"]["3"]
    pts = {}
    for h in hist:
        rl = dict(enumerate(h["rl"]))
        sar = dict(enumerate(h["sarops"]))
        pts[h["step"]] = ratios(rl, sar)
    from rl_search.scripts.real_map.vehicle_web import RUNS
    for d in RUNS.glob(f"eval-{split}-s3-k0-rl-{run}-*"):            # checkpoints scored afterwards (eval_rl --label <run>-<step>)
        tail = d.name.rsplit("-", 1)[1]
        if not tail.isdigit() or int(tail) in pts:
            continue
        rl, key = flight_set(split, f"{run}-{tail}")
        if len(rl) >= 3 and key:
            pts[int(tail)] = ratios(rl, sarops_of(split, key))
    steps = sorted(pts)
    return np.array(steps)/1e6, {t: np.array([pts[s].get(t, np.nan) for s in steps]) for t, _ in TYPES}


def main():
    warnings.filterwarnings("ignore")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, help="training run tags (folders of --root)")
    ap.add_argument("--names", nargs="*", default=None, help="legend names of the runs (default: the tags)")
    ap.add_argument("--split", default="nval", help="validation split of the runs' evaluations")
    ap.add_argument("--ref", action="append", default=[], help='reference line "name=<web run label>" scored on --split (repeatable)')
    ap.add_argument("--root", type=Path, default=Path("rl_search/data/rl_real"))
    ap.add_argument("--title", default=None)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    plt.rcParams["font.family"] = ["AppleGothic", "NanumGothic", "Noto Sans CJK KR", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False
    INK, MUTED, GRID, BG = "#1f1f1e", "#6b6b67", "#e4e3de", "#fcfcfb"
    names = args.names or args.runs
    data = {n: run_points(r, args.split, args.root) for n, r in zip(names, args.runs)}
    refs = {}
    for spec in args.ref:
        name, label = spec.split("=", 1)
        rl, key = flight_set(args.split, label)
        if rl and key:
            refs[name] = ratios(rl, sarops_of(args.split, key))
    fig, axs = plt.subplots(2, 2, figsize=(13.5, 8.6), facecolor=BG, sharex=True)
    xmax = max([x[-1] for x, _ in data.values() if len(x)] + [1.0])*1.08
    for ax, (key, title) in zip(axs.flat, TYPES):
        ax.set_facecolor(BG); ax.grid(color=GRID, lw=.6); [sp.set_color(GRID) for sp in ax.spines.values()]
        ax.tick_params(colors=MUTED, labelsize=9)
        ax.axhline(1, color=MUTED, ls="--", lw=1)
        ax.text(0.99, 1.0, "SAROPS = 1", transform=ax.get_yaxis_transform(), ha="right", va="bottom", fontsize=8, color=MUTED)
        for j, (name, v) in enumerate(refs.items()):
            if key in v:
                ax.axhline(v[key], color=COLORS[(len(data) + j) % len(COLORS)], ls=":", lw=1.4)
                ax.text(0.01, v[key], f"{name} {v[key]:.2f}", transform=ax.get_yaxis_transform(), va="bottom", fontsize=8.5,
                        color=COLORS[(len(data) + j) % len(COLORS)])
        for j, (name, (x, v)) in enumerate(data.items()):
            y = v[key]
            ax.plot(x, y, color=COLORS[j % len(COLORS)], lw=1.8, marker="os^"[j % 3], ms=4, label=name)
            if len(x):
                ax.annotate(f"{name} {y[-1]:.2f}", (x[-1], y[-1]), xytext=(6, 8 - 16*j), textcoords="offset points", va="center",
                            fontsize=9, color=COLORS[j % len(COLORS)])
        ax.set_title(f"{title}", loc="left", color=INK, fontsize=11)
        ax.set_xlim(0, xmax)
    for ax in axs[1]:
        ax.set_xlabel("학습 스텝 (백만)", color=MUTED, fontsize=9)
    for ax in axs[:, 0]:
        ax.set_ylabel("SAROPS 대비 비율 (hold-out)", color=MUTED, fontsize=9)
    axs[0, 0].legend(fontsize=9, frameon=False, loc="lower right")
    fig.suptitle(args.title or f"검증 {args.split}: " + " vs ".join(names), x=.01, ha="left", color=INK, fontsize=12)
    fig.tight_layout(); fig.savefig(args.out, dpi=115, facecolor=BG); print("saved", args.out)
    for name, (x, v) in data.items():
        print(name, " ".join(f"{s:.1f}M:{a:.3f}" for s, a in zip(x, v["all"])))
    for name, v in refs.items():
        print(name, {k: round(float(a), 3) for k, a in v.items()})


if __name__ == "__main__":
    main()

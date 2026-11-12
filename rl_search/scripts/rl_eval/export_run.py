"""One zip with everything needed to judge a training run elsewhere (user 2026-11-12: runs on a local PC that is not connected here and whose results are hard to take
out, so the essentials go into one file): settings, the validation history, all TensorBoard scalars as CSV, the evaluation results
(``eval_<split>_<label>.json`` of eval_rl), flight-shape numbers of the saved validation flights, figures, a summary in Markdown and,
by default, the best and the last model.

    python -m rl_search.scripts.rl_eval.export_run --run amboy_r7
    python -m rl_search.scripts.rl_eval.export_run --run amboy_r7 --models none      # without the weights (about 1-3 MB)
Writes rl_search/data/export/<run>_<date-time>/ and the same as a .zip next to it.
"""
from __future__ import annotations

import argparse
import base64
import csv
import gzip
import json
import platform
import re
import shutil
import time
import warnings
from pathlib import Path

import numpy as np

RUN_DIR = Path("rl_search/data/rl_real")
TYPES = ("road", "mixed", "desert")
TN = {"road": "도로형", "mixed": "혼합형", "desert": "사막형"}


def tb_scalars(run: Path) -> list[tuple]:
    """(source, tag, step, value) of every scalar under tb/ and tb_eval/."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    rows = []
    for d in sorted({p.parent for p in run.glob("tb*/**/events.out.tfevents.*")}):
        ea = EventAccumulator(str(d), size_guidance={"scalars": 0})
        ea.Reload()
        src = str(d.relative_to(run))
        for tag in ea.Tags()["scalars"]:
            rows += [(src, tag, int(e.step), float(e.value)) for e in ea.Scalars(tag)]
    return rows


def flight_shape(tag: str, rate_max_dps: float = 1.49) -> list[dict]:
    """Turning of the saved validation flights of this run (web runs ``eval-<split>-s3-k<k>-rl-<tag>-<step>``): minutes faster than the
    A turn limit, full circles (360 deg one way), minutes above 5 deg/s, minutes within 1 km of the map edge, and the hold-out score."""
    from rl_search.scripts.real_map.vehicle_web import RUNS
    out = []
    for d in sorted(RUNS.glob(f"eval-*-rl-{tag}-*")):
        m = re.match(rf"eval-(\w+)-s3-k(\d+)-rl-{re.escape(tag)}-(\d+)$", d.name)
        if not m or not (d/"run.json.gz").exists():
            continue
        r = json.load(gzip.open(d/"run.json.gz", "rt"))
        meta, u = r["meta"], r["opt"]["uavs"][0]
        t = np.frombuffer(base64.b64decode(u["log_t"]), np.float32).astype(float)
        xyy = np.frombuffer(base64.b64decode(u["log_xyyaw"]), np.float32).reshape(-1, 3).astype(float)
        if len(t) < 3:
            continue
        psi = np.unwrap(xyy[:, 2])
        w = np.degrees(np.gradient(psi, t))
        dt = np.gradient(t)
        s = np.sign(w)*(np.abs(w) > 1.0)
        loops, i = 0, 0
        while i < len(s):
            if s[i] == 0:
                i += 1
                continue
            j = i
            while j + 1 < len(s) and s[j + 1] == s[i]:
                j += 1
            loops += int(abs(np.degrees(psi[j] - psi[i]))//360)
            i = j + 1
        edge = 30012.5 - np.maximum(np.abs(xyy[:, 0]), np.abs(xyy[:, 1]))
        h = meta.get("holdout", {})
        out.append(dict(split=m.group(1), k=int(m.group(2)), step=int(m.group(3)), typ=meta.get("case_info", {}).get("typ", ""),
                        noA_min=float((dt*(np.abs(w) > rate_max_dps)).sum()/60), loops=loops, tight_min=float((dt*(np.abs(w) > 5)).sum()/60),
                        edge1km_min=float((dt*(edge < 1000)).sum()/60), pos=h.get("pos"), A=h.get("A"), total=h.get("total")))
    return out


def figures(out: Path, hist: list[dict], tb: list[tuple], shape: list[dict], title: str) -> None:
    import logging
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    have = {f.name for f in font_manager.fontManager.ttflist}         # the first Korean font this machine has (Linux: apt fonts-nanum)
    kr = next((n for n in ("AppleGothic", "NanumGothic", "Noto Sans CJK KR", "Noto Sans KR", "Malgun Gothic") if n in have), None)
    plt.rcParams["font.family"] = [kr, "sans-serif"] if kr else ["sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False
    INK, MUTED, GRID, BG = "#1f1f1e", "#6b6b67", "#e4e3de", "#fcfcfb"
    TC = {"road": "#2a78d6", "mixed": "#eb6834", "desert": "#1baf7a", "all": "#1f1f1e"}
    fig, axs = plt.subplots(2, 2, figsize=(13, 8.4), facecolor=BG)
    for ax in axs.ravel():
        ax.set_facecolor(BG)
        ax.grid(color=GRID, lw=.6)
        for sp in ax.spines.values():
            sp.set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=9)
        ax.set_xlabel("학습 스텝 (백만)", color=MUTED, fontsize=9)

    def series(src, tag):
        a = np.array([(s, v) for so, t, s, v in tb if so.startswith(src) and t == tag])
        return (a[:, 0]/1e6, a[:, 1]) if len(a) else (np.zeros(0), np.zeros(0))
    ax = axs[0, 0]
    x, y = series("tb/", "episode/total")
    if len(x):
        ax.plot(x, y, color=TC["road"], lw=1.0, label="학습 점수 (최근 100개 비행 평균)")
        k = 33
        ax.plot(x, np.array([y[max(0, i - k):i + k + 1].mean() for i in range(len(y))]), color=INK, lw=2, label="추세")
    ax.set_title("(가) 학습 점수 A + w POS", loc="left", color=INK, fontsize=10.5)
    ax.legend(fontsize=8, frameon=False)
    ax = axs[0, 1]
    if hist:
        st = np.array([h["step"] for h in hist])/1e6
        ax.plot(st, [h["ratio"] for h in hist], color=TC["all"], lw=2, marker="o", ms=4, label="전체")
        for t in TYPES:
            v = [h.get("type_ratio", {}).get(t, np.nan) for h in hist]
            ax.plot(st, v, color=TC[t], lw=1.2, marker="o", ms=3, label=TN[t])
        ax.axhline(1.0, color=MUTED, lw=1, ls="--")
    ax.set_title("(나) 검증: SAROPS 대비 비율", loc="left", color=INK, fontsize=10.5)
    ax.legend(fontsize=8, frameon=False)
    ax = axs[1, 0]
    x, y = series("tb/", "train/approx_kl")
    if len(x):
        ax.plot(x, y, color=TC["desert"], lw=.7, label="업데이트마다 정책 변화량 (KL)")
        ax.axhline(0.03, color=TC["mixed"], lw=1, ls="--", label="조기 중단 0.03")
    x2, y2 = series("tb/", "train/explained_variance")
    ax.set_title("(다) 업데이트 크기 (KL)", loc="left", color=INK, fontsize=10.5)
    ax.legend(fontsize=8, frameon=False)
    ax = axs[1, 1]
    if shape:
        steps = sorted({r["step"] for r in shape})
        for t in TYPES + ("all",):
            v = [np.mean([r["noA_min"] for r in shape if r["step"] == s and (t == "all" or r["typ"] == t)] or [np.nan]) for s in steps]
            ax.plot(np.array(steps)/1e6, v, color=TC[t], lw=2 if t == "all" else 1.2, marker="o", ms=4, label="전체" if t == "all" else TN[t])
    ax.set_title("(라) 검증 비행: A 인정 한계(1.49°/s)보다 빨리 돈 분", loc="left", color=INK, fontsize=10.5)
    ax.legend(fontsize=8, frameon=False)
    fig.suptitle(title, x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out/"curves.png", dpi=110, facecolor=BG)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="tag under rl_search/data/rl_real")
    ap.add_argument("--models", default="best,last", help="best,last / best / last / none")
    ap.add_argument("--out", type=Path, default=Path("rl_search/data/export"))
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    run = RUN_DIR/args.run
    if not run.is_dir():
        raise SystemExit(f"no run folder {run}")
    out = args.out/f"{args.run}_{time.strftime('%Y%m%d-%H%M')}"
    out.mkdir(parents=True, exist_ok=True)
    for f in ("config.json", "state.json", "type_weights.json"):
        if (run/f).exists():
            shutil.copy2(run/f, out/f)
    cfg = json.loads((run/"config.json").read_text()) if (run/"config.json").exists() else {}
    state = json.loads((run/"state.json").read_text()) if (run/"state.json").exists() else {}
    hist = [h for st in sorted(state.get("history", {})) for h in state["history"][st]]
    with open(out/"eval_history.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "ratio", "rl_mean", "sarops_mean"] + [f"ratio_{t}" for t in TYPES] + [f"rl_k{k}" for k in range(len(hist[0]["rl"]) if hist else 0)])
        for h in hist:
            w.writerow([h["step"], h.get("ratio"), np.mean(h["rl"]), np.mean(h["sarops"]) if h.get("sarops") else ""]
                       + [h.get("type_ratio", {}).get(t, "") for t in TYPES] + list(h["rl"]))
    tb = tb_scalars(run)
    with open(out/"tb_scalars.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source", "tag", "step", "value"])
        w.writerows(tb)
    evals = sorted(run.glob("ckpt/eval_*.json")) + sorted(run.glob("eval_*.json"))
    for e in evals:
        shutil.copy2(e, out/e.name)
    tag_label = f"{args.run}" if (cfg.get("init") or cfg.get("multi")) else ""
    shape = flight_shape(tag_label or args.run)
    with open(out/"flight_shape.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["split", "k", "step", "typ", "noA_min", "loops", "tight_min", "edge1km_min", "pos", "A", "total"])
        w.writeheader()
        w.writerows(shape)
    figures(out, hist, tb, shape, f"{args.run} · 내보낸 시각 {time.strftime('%Y-%m-%d %H:%M')}")
    # models
    want = set(args.models.split(",")) - {"none", ""}
    ck = sorted(run.glob("ckpt/model_*.zip"), key=lambda p: int(re.findall(r"model_(\d+)", p.name)[0]))
    if "best" in want and (run/"best_stage3.zip").exists():
        shutil.copy2(run/"best_stage3.zip", out/"best_stage3.zip")
    if "last" in want and ck:
        shutil.copy2(ck[-1], out/ck[-1].name)
    # summary
    L = [f"# {args.run} 결과 요약", "", f"- 내보낸 시각: {time.strftime('%Y-%m-%d %H:%M')}, 기계: {platform.node()} ({platform.platform()}, "
         f"{platform.processor() or platform.machine()})", f"- 설정: " + ", ".join(f"{k}={cfg[k]}" for k in sorted(cfg) if k not in ("out",)), ""]
    if hist:
        last = hist[-1]
        r3 = np.mean([h["ratio"] for h in hist[-3:] if h.get("ratio") is not None]) if any(h.get("ratio") for h in hist[-3:]) else float("nan")
        best = max((h for h in hist if h.get("ratio") is not None), key=lambda h: h["ratio"], default=None)
        L += ["## 검증 (SAROPS 대비 비율)", "", f"- 마지막 평가 {last['step']:,} 스텝: {last.get('ratio') or float('nan'):.3f}, 최근 3번 평균 {r3:.3f}",
              f"- 최고: {best['step']:,} 스텝 {best['ratio']:.3f}" if best else "- 최고: 없음",
              "- 유형별(마지막): " + ", ".join(f"{TN[t]} {last.get('type_ratio', {}).get(t, float('nan')):.3f}" for t in TYPES), "",
              "| 스텝 | 전체 | 도로형 | 혼합형 | 사막형 |", "|---|---|---|---|---|"]
        for h in hist:
            tr = h.get("type_ratio", {})
            ratio = "-" if h.get("ratio") is None else format(h["ratio"], ".3f")
            L.append(f"| {h['step']:,} | {ratio} | " + " | ".join(format(tr.get(t, float("nan")), ".3f") for t in TYPES) + " |")
        L.append("")
    for e in evals:
        try:
            rows = json.loads(e.read_text())
            have = [r for r in rows if r.get("sarops")]
            rl = np.array([r["holdout"]["total"] for r in have])
            sa = np.array([r["sarops"]["total"] for r in have])
            d = rl - sa
            L += [f"## {e.name}", "", f"- 사례 {len(rows)}개, SAROPS 기준값 있는 {len(have)}개: RL {rl.mean():.3f}, SAROPS {sa.mean():.3f}, "
                  f"짝지은 차 {d.mean():+.3f} ± {d.std(ddof=1)/np.sqrt(max(len(d), 2)):.3f} (표준오차), RL이 높은 사례 {(d > 0).sum()}/{len(d)}"
                  if len(have) else f"- 사례 {len(rows)}개 (SAROPS 기준값 없음)", ""]
        except (ValueError, KeyError, TypeError):
            pass
    if shape:
        L += ["## 검증 비행 모양 (체크포인트별 평균)", "", "| 스텝 | A 한계보다 빨리 돈 분 | 360° 회전 | 급선회 분 | 가장자리 1 km 안 분 |", "|---|---|---|---|---|"]
        for s in sorted({r["step"] for r in shape}):
            rr = [r for r in shape if r["step"] == s]
            L.append(f"| {s:,} | {np.mean([r['noA_min'] for r in rr]):.1f} | {np.mean([r['loops'] for r in rr]):.1f} | "
                     f"{np.mean([r['tight_min'] for r in rr]):.1f} | {np.mean([r['edge1km_min'] for r in rr]):.1f} |")
        L.append("")
    L += ["## 들어 있는 파일", "", "- `config.json`, `state.json`, `type_weights.json`: 학습 설정과 검증 기록 원본",
          "- `eval_history.csv`: 검증마다 비율, 유형별 비율, 사례별 점수", "- `tb_scalars.csv`: TensorBoard 값 전부(학습 곡선, KL, 유형 비율 등)",
          "- `flight_shape.csv`: 저장된 검증 비행의 선회 지표", "- `eval_*.json`: eval_rl 결과(시험 평가 등)", "- `curves.png`: 위 내용 그림",
          "- `best_stage3.zip`, `model_<스텝>.zip`: 모델 가중치(PPO.load로 불러옴)"]
    (out/"summary.md").write_text("\n".join(L), encoding="utf-8")
    z = shutil.make_archive(str(out), "zip", root_dir=out.parent, base_dir=out.name)
    size = Path(z).stat().st_size/1e6
    print(f"exported {args.run}: {out} ({len(hist)} evaluations, {len(tb)} scalars, {len(shape)} flights, {len(evals)} eval files) -> {z} ({size:.1f} MB)")


if __name__ == "__main__":
    main()

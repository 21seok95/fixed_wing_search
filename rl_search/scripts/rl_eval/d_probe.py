"""The Chungju edge / visibility probe (user 2026-11-17, docs/rl_experiments.md) for any model, without the web viewer: the user's 15
Chungju web flights (rl_search/data/real/chungju/web_cases_20261117.json: the page's target settings, LKP, heading, launch point), each
flown as the web viewer flies an RL model (web_rl._run: same scenario, deterministic flight, hold-out replay) in four variants:

  A - as the model was trained (off the map: visibility 0 in the observation; particles that leave the map are dropped)
  B - particles that leave the map go on straight (VehicleConfig.exit_straight) and stay in the belief
  C - B, and the ground off the map is open: visibility 1 in the observation and in the detection (setting off_map_vis)
  D - C, and the policy also sees the ground inside the map as open (visibility channel 1 everywhere; the detection inside keeps the
      real Chungju visibility, so POS is what the real ground gives)

Results per model and variant in rl_search/data/real/chungju/d_probe/<run>_<variant>.json (POS, path, particles at the end) and, with
--web, as web runs too. The cases file holds the Mac's R14 road (3.2M) POS per variant: flying r14_road reproduces them exactly when the
environment is the same.

    python -m rl_search.scripts.rl_eval.d_probe run --model rl_search/data/rl_real/r13_2map/ckpt/model_6000000.zip --variant A   (B, C, D)
    python -m rl_search.scripts.rl_eval.d_probe figure --run r13_2map
    python -m rl_search.scripts.rl_eval.d_probe export          (Mac: write the cases file from the web runs of 2026-11-17)
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np

CASES = Path("rl_search/data/real/chungju/web_cases_20261117.json")
OUT = Path("rl_search/data/real/chungju/d_probe")
VARIANTS = ("A", "B", "C", "D")
VN = {"A": "A 학습 그대로", "B": "B 지도 밖 직진", "C": "C 직진 + 밖 가시도 1", "D": "D 직진 + 안팎 가시도 1"}


class _Q:
    def put(self, m):
        pass


def export():
    """Mac: the cases (the user's Chungju RL web flights of 2026-11-17) and R14 road's POS per variant from the web runs of the probes."""
    from rl_search.scripts.rl_eval.circling_cause import web_runs
    runs = web_runs()
    orig = [m for m in runs if "시험" not in m["label"]]
    tags = {"B": "시험: 지도 밖 직진 (", "C": "시험: 지도 밖 직진 + 밖 가시도 1 (", "D": "시험 D: 직진 + 안팎 가시도 1 ("}
    cases = []
    for i, m in enumerate(orig):
        ref = {"A": m["final"]}
        for v, tag in tags.items():
            hit = [x for x in runs if tag in x["label"] and f"(원래 {m['id'][9:15]})" in x["label"]]
            if hit:
                ref[v] = hit[-1]["final"]
        cases.append(dict(i=i + 1, id=m["id"], label=m["label"], view=m["view"], config=m["config"], launch_min=m.get("launch_min", 0.0),
                          r14_road_pos=ref))
    CASES.write_text(json.dumps(cases, ensure_ascii=False, indent=1))
    print(f"{len(cases)} cases -> {CASES}")


def run(model: str, variant: str, web: bool):
    warnings.filterwarnings("ignore")
    import rl_search.scripts.rl_eval.signal_check as sc
    from rl_search.rl_real import RealRLEnv
    from rl_search.scripts.real_map import web_rl
    base = sc.settings_of
    if variant in ("C", "D"):
        sc.settings_of = lambda cfg: base(cfg) | dict(off_map_vis=1.0)
    if variant == "D":
        orig = RealRLEnv.observation

        def observation(self):
            o = orig(self)
            for k in o:
                if k != "vec":
                    o[k][4] = 1.0
            return o
        RealRLEnv.observation = observation
    run_name = web_rl.run_dir(Path(model)).name
    cases = json.loads(CASES.read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    out = []
    stamp = time.strftime("%Y%m%d-%H%M%S")
    for c in cases:
        cfg = dict(c["config"], exit_straight=variant != "A")
        p = dict(view=c["view"], config=cfg, model=model, launch_min=c.get("launch_min", 0.0), kind="rl")
        t0 = time.perf_counter()
        data = web_rl._run(p, _Q())
        sec = time.perf_counter() - t0
        o, res = data["opt"], data["res"]
        u = o["uavs"][0]
        import base64
        xy = np.frombuffer(base64.b64decode(u["xy"]), np.float32).reshape(-1, 2)
        tt = np.frombuffer(base64.b64decode(u["tt"]), np.float32)
        T, N = res["T"], res["N"]
        P = np.frombuffer(base64.b64decode(res["pos"]), "<i2").reshape(T, N, 2).astype(np.float32)*res["pos_unit_m"]
        W = np.frombuffer(base64.b64decode(o["weight"]), np.uint8).reshape(o["T"], N)
        k = min(o["T"], T) - 1
        keep = np.searchsorted(tt, np.arange(0, float(tt[-1]) + 1, 5.0)).clip(0, len(tt) - 1)
        ref = c.get("r14_road_pos", {}).get(variant)
        out.append(dict(i=c["i"], pos=float(o["final"]), r14_road_mac=ref, xy=xy[keep].round(1).tolist(), tt=tt[keep].round(1).tolist(),
                        p_end=P[k, ::2].round(0).tolist(), w_end=(W[k, ::2]/255).round(3).tolist(), lkp=o["lkp"], start=o["start"], sec=round(sec, 1)))
        print(f"{run_name} {variant} {c['i']}: POS {o['final']:.3f}" + (f" (Mac R14 road {ref:.3f})" if ref is not None else "") + f", {sec:.0f} s",
              flush=True)
        if web:
            from rl_search.scripts.real_map.vehicle_web import run_meta, save_run
            data["opt"]["sec"] = round(sec, 1)
            extra = data.pop("meta_extra", {})
            rid = f"{stamp}-{dict(A=500, B=600, C=700, D=800)[variant] + c['i']:03d}"
            meta = run_meta(data["opt"], rid, f"{c['label'].split(' · RL')[0]} · {run_name} · {VN[variant]}", time.strftime("%m-%d %H:%M"),
                            c["view"], cfg, group="web", **extra)
            save_run(rid, data, meta)
    (OUT/f"{run_name}_{variant}.json").write_text(json.dumps(out))


def metrics(xy, tt, half=30000.0):
    xy, tt = np.asarray(xy, float), np.asarray(tt, float)
    cum = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(xy, axis=0).T))])
    ratios = []
    for j in range(len(xy)):
        e = np.searchsorted(tt, tt[j] + 600.0)
        if e >= len(xy):
            break
        ratios.append(np.hypot(*(xy[e] - xy[j]))/max(cum[e] - cum[j], 1.0))
    return dict(edge=float((np.abs(xy).max(axis=1) > half - 2000).mean()), off=float((np.abs(xy).max(axis=1) > half).mean()),
                loiter=float(np.mean(np.array(ratios) < 0.2)) if ratios else 0.0)


def figure(run_name: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from rl_search.rl_real import preload
    from rl_search.scripts.real_map.vehicle_tuner import map_image
    plt.rcParams["font.family"] = ["AppleGothic", "NanumGothic", "Noto Sans CJK KR", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False
    D = {v: json.loads((OUT/f"{run_name}_{v}.json").read_text()) for v in VARIANTS if (OUT/f"{run_name}_{v}.json").exists()}
    img, ext = map_image(preload("chungju")["net"], 1600)
    vs = list(D)
    n = len(next(iter(D.values())))
    fig, axs = plt.subplots(n, len(vs), figsize=(5*len(vs), 5*n), facecolor="#fcfcfb", squeeze=False)
    rows = []
    for i in range(n):
        allp = np.vstack([np.asarray(D[v][i]["xy"])/1000 for v in vs])
        lo, hi = np.percentile(allp, 1, axis=0), np.percentile(allp, 99, axis=0)
        c, h = (lo + hi)/2, max((hi - lo).max()/2 + 2, 6)
        row = [i + 1]
        for j, v in enumerate(vs):
            r = D[v][i]
            ax = axs[i, j]
            xy = np.asarray(r["xy"])/1000
            ax.imshow(img, extent=ext, origin="upper", alpha=0.8)
            P, w = np.asarray(r["p_end"])/1000, np.asarray(r["w_end"])
            ax.scatter(P[:, 0], P[:, 1], s=1 + 10*w, c=w, cmap="autumn_r", vmin=0, vmax=1, lw=0, alpha=0.6)
            m3 = len(xy)//3
            for a, b, col in ((0, m3, "#1f3a93"), (m3, 2*m3, "#7b2cbf"), (2*m3, len(xy), "#000000")):
                ax.plot(xy[a:b + 1, 0], xy[a:b + 1, 1], color=col, lw=1.0)
            ax.plot([-30, 30, 30, -30, -30], [-30, -30, 30, 30, -30], color="#d6336c", lw=0.8, ls="--")
            ax.plot(*(np.asarray(r["lkp"])/1000), marker="*", ms=12, color="#d6336c", mec="white")
            ax.set_xlim(c[0] - h, c[0] + h); ax.set_ylim(c[1] - h, c[1] + h); ax.set_aspect("equal")
            mt = metrics(r["xy"], r["tt"])
            ax.set_title(f"{i + 1}번 {VN[v]}: POS {r['pos']:.2f}\n가장자리 {100*mt['edge']:.0f}%, 지도 밖 {100*mt['off']:.0f}%, 맴돎 {100*mt['loiter']:.0f}%",
                         loc="left", fontsize=10)
            ax.tick_params(labelsize=7)
            row.append((r["pos"], mt))
        rows.append(row)
    fig.suptitle(f"충주 15개, {run_name}: 경로 색 = 앞 1/3 파랑, 가운데 보라, 끝 1/3 검정; 맴돎 = 10분 이동이 날아간 거리의 20% 미만", x=0.01, ha="left", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.995))
    fig.savefig(OUT/f"{run_name}_paths.png", dpi=55, facecolor="#fcfcfb")
    L = [f"# 충주 시험 ({run_name})", "", "| 번호 | " + " | ".join(f"{VN[v]} POS / 맴돎 / 가장자리" for v in vs) + " |", "|---|" + "---|"*len(vs)]
    for row in rows:
        L.append(f"| {row[0]} | " + " | ".join(f"{p:.3f} / {100*m['loiter']:.0f}% / {100*m['edge']:.0f}%" for p, m in row[1:]) + " |")
    for j, v in enumerate(vs):
        p = [row[1 + j][0] for row in rows[:13]]
        lt = [row[1 + j][1]["loiter"] for row in rows[:13]]
        L.append(f"\n{VN[v]}: 1~13번 평균 POS {np.mean(p):.3f}, 맴돎 {100*np.mean(lt):.0f}%")
    (OUT/f"{run_name}_table.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=("run", "figure", "export"))
    ap.add_argument("--model", default="rl_search/data/rl_real/r13_2map/ckpt/model_6000000.zip")
    ap.add_argument("--variant", choices=VARIANTS, default="A")
    ap.add_argument("--run", default="r13_2map", help="figure: the run name (the model's training folder)")
    ap.add_argument("--web", action="store_true", help="also save every flight as a web run (the viewer lists it)")
    a = ap.parse_args()
    if a.what == "export":
        export()
    elif a.what == "run":
        run(a.model, a.variant, a.web)
    else:
        figure(a.run)


if __name__ == "__main__":
    main()

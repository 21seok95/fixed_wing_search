"""Does the camera keep the particles in view? (user 2026-11-18: Chungju web flights of R14 road with particles that do not branch;
case 1 particles leave the map, case 2 they stay and late in the flight the UAV searches beside the road with the particles out of the
image). Each web run of the given day on chungju is flown again as the web viewer flies it, the scenario unchanged, with the policy
shown visibility 1 inside and off the map (the detection keeps the real Chungju visibility inside, 1 off the map); saved as a web run.

Per logged second of every flight (the original and the probe): the remaining probability inside the camera footprint (the image
quadrilateral on the ground, particles at the nearest saved time with their remaining weight at that time) and the distance from the
footprint centre to the nearest particle with weight left.

    python -m rl_search.scripts.rl_eval.fov_probe run --day 20261118
    python -m rl_search.scripts.rl_eval.fov_probe figure --day 20261118
"""
from __future__ import annotations

import argparse
import base64
import gzip
import json
import time
import warnings
from pathlib import Path

import numpy as np

RUNS = Path("rl_search/data/real/chungju/web_runs")
OUT = Path("rl_search/data/real/amboy/rl_eval/report")
TAG = "시험: 정책에게 안팎 가시도 1"


class _Q:
    def put(self, m):
        pass


def originals(day):
    out = []
    for d in sorted(RUNS.glob(f"{day}-*")):
        m = json.loads((d/"meta.json").read_text())
        if (m.get("view") or {}).get("region") == "chungju" and m.get("model") and "시험" not in m["label"]:
            out.append(m)
    return out


def run(day):
    warnings.filterwarnings("ignore")
    import rl_search.scripts.rl_eval.signal_check as sc
    from rl_search.rl_real import RealRLEnv
    from rl_search.scripts.real_map import web_rl
    from rl_search.scripts.real_map.vehicle_web import run_meta, save_run
    base = sc.settings_of
    sc.settings_of = lambda cfg: base(cfg) | dict(off_map_vis=1.0)
    orig = RealRLEnv.observation

    def observation(self):
        o = orig(self)
        for k in o:
            if k != "vec":
                o[k][4] = 1.0
        return o
    RealRLEnv.observation = observation
    stamp = time.strftime("%Y%m%d-%H%M%S")
    for i, m in enumerate(originals(day)):
        p = dict(view=m["view"], config=m["config"], model=m["model"], launch_min=m.get("launch_min", 0.0), kind="rl")
        t0 = time.perf_counter()
        data = web_rl._run(p, _Q())
        data["opt"]["sec"] = round(time.perf_counter() - t0, 1)
        extra = data.pop("meta_extra", {})
        rid = f"{stamp}-{950 + i:03d}"
        meta = run_meta(data["opt"], rid, f"{m['label']} · {TAG} (원래 {m['id'][9:15]})", time.strftime("%m-%d %H:%M"), m["view"], m["config"],
                        group="web", **extra)
        save_run(rid, data, meta)
        print(f"{m['id']} -> {rid}: POS {data['opt']['final']:.3f} (original {m['final']:.3f})", flush=True)


def fov_series(rid):
    from matplotlib.path import Path as MPath
    d = json.loads(gzip.decompress((RUNS/rid/"run.json.gz").read_bytes()))
    o, res = d["opt"], d["res"]
    dec = lambda s, dt: np.frombuffer(base64.b64decode(s), dtype=dt)
    T, N = res["T"], res["N"]
    P = dec(res["pos"], "<i2").reshape(T, N, 2).astype(float)*res["pos_unit_m"]
    W = dec(o["weight"], np.uint8).reshape(o["T"], N).astype(float)/255
    u = o["uavs"][0]
    lt = dec(u["log_t"], np.float32).astype(float)
    quad = dec(u["quad"], np.float32).reshape(-1, 4, 2).astype(float)
    dt = float(res["dt_s"])
    inside_p, near_m = [], []
    for t, q in zip(lt, quad):
        k = min(int(round(t/dt)), T - 1, o["T"] - 1)
        w = W[k]/N
        live = w > 1e-6
        Pk = P[k][live]
        if not len(Pk):
            inside_p.append(0.0); near_m.append(np.nan); continue
        inn = MPath(q).contains_points(Pk)
        inside_p.append(float(w[live][inn].sum()))
        c = q.mean(axis=0)
        near_m.append(float(np.hypot(*(Pk - c).T).min()))
    xy = dec(u["xy"], np.float32).reshape(-1, 2).astype(float)
    return dict(t=lt, p_in=np.array(inside_p), near=np.array(near_m), quad=quad, xy=xy, final=o["final"], P=P, W=W, dt=dt, N=N)


def figure(day):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from rl_search.rl_real import preload
    from rl_search.scripts.real_map.vehicle_tuner import map_image
    plt.rcParams["font.family"] = ["AppleGothic", "NanumGothic", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False
    INK, MUTED, GRID, BG = "#1f1f1e", "#6b6b67", "#e4e3de", "#fcfcfb"
    origs = originals(day)
    probes = {}
    for d in sorted(RUNS.glob("*")):
        f = d/"meta.json"
        if f.exists():
            m = json.loads(f.read_text())
            if TAG in m.get("label", ""):
                probes[m["label"].split("(원래 ")[1][:6]] = m["id"]
    img, ext = map_image(preload("chungju")["net"], 1600)
    n = len(origs)
    fig, axs = plt.subplots(n, 3, figsize=(24, 7.2*n), facecolor=BG, squeeze=False, gridspec_kw=dict(width_ratios=[1, 1, 1.25]))
    lines = []
    for i, m in enumerate(origs):
        runs = [("원래", fov_series(m["id"]))]
        if m["id"][9:15] in probes:
            runs.append(("정책에게 가시도 1", fov_series(probes[m["id"][9:15]])))
        allxy = np.vstack([r["xy"][len(r["xy"])//2:] for _, r in runs])/1000          # zoom on the late half of the flights
        lo, hi = np.percentile(allxy, 1, axis=0), np.percentile(allxy, 99, axis=0)
        cc, h = (lo + hi)/2, max((hi - lo).max()/2 + 0.5, 1.5)
        for j, (nm, r) in enumerate(runs):
            ax = axs[i, j]
            ax.imshow(img, extent=ext, origin="upper", alpha=0.85)
            k_end = min(len(r["P"]), len(r["W"])) - 1
            # particles at 40 min (remaining weight), the late half's footprints coloured by whether they held any remaining probability
            k40 = min(int(round(2400/r["dt"])), k_end)
            w = r["W"][k40]
            ax.scatter(r["P"][k40, :, 0]/1000, r["P"][k40, :, 1]/1000, s=1 + 8*w, c=w, cmap="autumn_r", vmin=0, vmax=1, lw=0, alpha=0.6, zorder=2)
            late = r["t"] >= r["t"][-1]/2
            for q, pin in zip(r["quad"][late][::2], r["p_in"][late][::2]):
                ax.fill(*(np.vstack([q, q[:1]])/1000).T, color="#1baf7a" if pin > 1e-4 else "#d6336c", alpha=0.25, lw=0, zorder=3)
            hx = r["xy"][len(r["xy"])//2:]/1000
            ax.plot(hx[:, 0], hx[:, 1], color="#1f3a93", lw=0.8, zorder=4)
            ax.set_xlim(cc[0] - h, cc[0] + h); ax.set_ylim(cc[1] - h, cc[1] + h); ax.set_aspect("equal")
            share = float((r["p_in"][late] > 1e-4).mean())
            ax.set_title(f"{m['id'][9:15]} {nm}: POS {r['final']:.2f} (후반 30분 확대)\n카메라 화면: 초록 = 남은 입자 있음, 빨강 = 없음 (있음 {100*share:.0f}%); 점 = 40분 입자",
                         loc="left", fontsize=10.5, color=INK)
            ax.tick_params(labelsize=7.5)
            lines.append((m["id"][9:15], nm, r["final"], share, float(np.nanmedian(r["near"][late]))))
        ax = axs[i, 2]
        for (nm, r), col in zip(runs, (INK, "#eb6834")):
            tm = r["t"]/60
            k = 60
            sm = np.convolve((r["p_in"] > 1e-4).astype(float), np.ones(k)/k, mode="same")
            ax.plot(tm, 100*sm, color=col, lw=1.8, label=f"{nm}")
        ax.set_xlabel("신고 뒤 분", fontsize=9, color=MUTED); ax.set_ylim(0, 105)
        ax.set_ylabel("화면에 남은 입자가 있던 시간 (%, 1분 이동 평균)", fontsize=9, color=MUTED)
        ax.set_title("카메라 화면 안에 남은 입자가 있었나 (1초마다)", loc="left", fontsize=10.5, color=INK)
        ax.legend(fontsize=8, frameon=False); ax.grid(color=GRID, lw=.5)
    for a_ in fig.axes:
        a_.set_facecolor(BG)
    fig.suptitle(f"충주 웹 비행 ({day}, R14 도로형): 원래 대 정책에게 가시도 1(실제 탐지는 진짜 충주). 카메라 화면이 길 위 입자를 따라가는가", x=0.01, ha="left",
                 fontsize=14, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(OUT/"fov_probe.png", dpi=62, facecolor=BG)
    for ln in lines:
        print(f"{ln[0]} {ln[1]}: POS {ln[2]:.3f}, late half seconds with remaining particles in the image {100*ln[3]:.0f}%, median nearest particle {ln[4]:.0f} m")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=("run", "figure"))
    ap.add_argument("--day", default="20261118")
    a = ap.parse_args()
    run(a.day) if a.what == "run" else figure(a.day)


if __name__ == "__main__":
    main()

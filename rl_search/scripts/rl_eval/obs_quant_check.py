"""Can the observation maps be stored in 16 or 8 bits? (user 2026-11-23: rollout memory and inter-process transfer grow with the number
of UAVs; 10 UAVs with float32 maps need about 14.6 GB of rollout storage.)  No training: the R15 model on the Jirisan validation cases.

1. Value range of every map channel (what an 8-bit store in [0, 1] would clip).
2. Open loop: on every decision of the float32 flight, the same observation rounded to float16 and to 8 bits (x -> round(clip(x, 0, 1)
   * 255) / 255, what an uint8 image store gives back): the change of the policy's mean action (degrees of heading change per 5 s) and
   of the value.
3. Closed loop: the whole flight with each rounding (deterministic policy), hold-out POS and how far the track drifts from float32's.
4. Cost: bytes per decision of the maps in each format, policy forward time float32 vs float16 (CPU, MPS).

    python -m rl_search.scripts.rl_eval.obs_quant_check --workers 3
Writes report/obs_quant_check.png and .json.
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from multiprocessing import get_context
from pathlib import Path

import numpy as np

OUT = Path("rl_search/data/real/amboy/rl_eval/report")
RUN = Path("rl_search/data/rl_real/r15_chungju")
MODES = ("f32", "f16", "u8")


def quant(obs: dict, mode: str) -> dict:
    if mode == "f32":
        return obs
    out = {}
    for k, v in obs.items():
        if k == "vec":                                      # the small vector stays float32
            out[k] = v
        elif mode == "f16":
            out[k] = v.astype(np.float16).astype(np.float32)
        else:
            out[k] = (np.round(np.clip(v, 0.0, 1.0)*255.0)/255.0).astype(np.float32)
    return out


def fly(job):
    model_path, k, mode = job
    warnings.filterwarnings("ignore")
    import torch
    from stable_baselines3.common.utils import obs_as_tensor
    from rl_search.rl_real_eval import _case, _env, _model
    from rl_search.scripts.rl_eval.sarops_ref import holdout_replay
    from rl_search.scripts.rl_eval.signal_check import settings_of
    torch.set_num_threads(1)
    s = settings_of(json.loads((RUN/"config.json").read_text()))
    case, scen, _, hold = _case("v2j", 3, k)
    model, env = _model(model_path), _env(s, case.region)
    e = env.env
    obs, _ = env.reset(options=case.options(scen))
    turn_deg = float(np.degrees(e.turn))
    acts, xy, rows, ranges = [], [], [], {}
    term = trunc = False
    while not (term or trunc):
        oq = quant(obs, mode)
        if mode == "f32":                                   # open loop on the float32 flight
            for kk, v in obs.items():
                if kk != "vec":
                    r = ranges.setdefault(kk, [np.inf, -np.inf, 0, 0])
                    r[0] = min(r[0], float(v.min())); r[1] = max(r[1], float(v.max()))
                    r[2] += int((v > 1.0).sum()); r[3] += v.size
            with torch.no_grad():
                res = {}
                for m in MODES:
                    t = obs_as_tensor({kk: vv[None] for kk, vv in quant(obs, m).items()}, model.device)
                    d = model.policy.get_distribution(t)
                    res[m] = (float(d.distribution.mean[0, 0]), float(model.policy.predict_values(t)[0, 0]))
            rows.append([res["f32"][0], res["f16"][0], res["u8"][0], res["f32"][1], res["f16"][1], res["u8"][1]])
        a, _ = model.predict(oq, deterministic=True)
        acts.append(float(np.ravel(a)[0]))
        obs, _, term, trunc, info = env.step(a)
        xy.append(e.plant.pose[:2].copy())
    pos_h, _, _, _ = holdout_replay(env.s, env.terr, case, hold, [acts])      # the env's full settings (as rl_real_eval)
    return dict(k=k, mode=mode, pos_belief=float(info["POS"]), pos_holdout=float(pos_h), xy=np.array(xy).tolist(), turn_deg=turn_deg,
                rows=rows, ranges={kk: v for kk, v in ranges.items()})


def cost(model_path):
    """Bytes of the maps per decision in each format, policy forward time float32 / float16 (CPU, MPS), batch 1 and 10."""
    import torch
    from stable_baselines3 import PPO
    from rl_search.rl_real_eval import _case, _env
    from rl_search.scripts.rl_eval.signal_check import settings_of
    s = settings_of(json.loads((RUN/"config.json").read_text()))
    case, scen, _, _ = _case("v2j", 3, 0)
    env = _env(s, case.region)
    obs, _ = env.reset(options=case.options(scen))
    n_map = sum(v.size for kk, v in obs.items() if kk != "vec")
    out = dict(map_values=int(n_map), bytes=dict(f32=4*n_map, f16=2*n_map, u8=n_map), timing={})
    for dev in ("cpu", "mps"):
        if dev == "mps" and not torch.backends.mps.is_available():
            continue
        model = PPO.load(model_path, device=dev)
        for dtype in (torch.float32, torch.float16):
            pol = model.policy.to(dev).to(dtype)
            for B in (1, 10):
                t = {kk: torch.as_tensor(np.repeat(v[None], B, 0), device=dev, dtype=dtype) for kk, v in obs.items()}
                key = f"{dev} {str(dtype).split('.')[-1]} batch {B}"
                try:
                    with torch.no_grad():
                        pol.get_distribution(t)
                except Exception as ex:                       # e.g. SB3's preprocessing casts inputs to float32
                    out["timing"][key] = f"error: {type(ex).__name__}"
                    continue
                with torch.no_grad():
                    for _ in range(5):
                        pol.get_distribution(t)
                    if dev == "mps":
                        torch.mps.synchronize()
                    t0 = time.perf_counter()
                    for _ in range(50):
                        pol.get_distribution(t)
                    if dev == "mps":
                        torch.mps.synchronize()
                out["timing"][key] = round((time.perf_counter() - t0)/50*1000, 2)
            pol.to(torch.float32)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=str(RUN/"ckpt/model_8000000.zip"))
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    jobs = [(a.model, k, m) for k in range(3) for m in MODES]
    with get_context("fork").Pool(a.workers, maxtasksperchild=1) as pool:
        res = pool.map(fly, jobs, chunksize=1)
    c = cost(a.model)
    summary = figure(res, c)
    (OUT/"obs_quant_check.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print(json.dumps(summary, ensure_ascii=False, indent=1)[:4000])


def figure(res, c):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["AppleGothic", "sans-serif"]; plt.rcParams["axes.unicode_minus"] = False
    INK, MUTED, GRID, BG = "#1f1f1e", "#6b6b67", "#e4e3de", "#fcfcfb"
    COL = dict(f32="#6b6b67", f16="#2a78d6", u8="#E69F00")
    NAME = dict(f32="32비트 (지금)", f16="16비트", u8="8비트")
    base = [r for r in res if r["mode"] == "f32"]
    R = np.concatenate([np.array(r["rows"]) for r in base])
    turn = base[0]["turn_deg"]
    da16, da8 = np.abs(R[:, 1] - R[:, 0])*turn, np.abs(R[:, 2] - R[:, 0])*turn
    dv16, dv8 = np.abs(R[:, 4] - R[:, 3]), np.abs(R[:, 5] - R[:, 3])
    rng = {}
    for r in base:
        for kk, (lo, hi, over, n) in r["ranges"].items():
            g = rng.setdefault(kk, [np.inf, -np.inf, 0, 0])
            g[0] = min(g[0], lo); g[1] = max(g[1], hi); g[2] += over; g[3] += n
    fig, ax = plt.subplots(2, 2, figsize=(16, 10.5), facecolor=BG)
    a0 = ax[0, 0]
    names = list(rng)
    a0.bar(range(len(names)), [rng[n][1] for n in names], color="#2a78d6", alpha=.8)
    a0.axhline(1.0, color="#d6336c", ls="--", lw=1)
    for i, n in enumerate(names):
        a0.text(i, rng[n][1] + .02, f"{rng[n][1]:.2f}\n1 넘는 값 {100*rng[n][2]/max(rng[n][3], 1):.3f}%", ha="center", fontsize=8.5, color=INK)
    a0.set_xticks(range(len(names)), [n.replace("m", "") + " m" for n in names], fontsize=9)
    a0.set_title("관측 지도의 최댓값 (지도별, 모든 채널; 빨간 선 = 8비트가 담는 상한 1)", loc="left", fontsize=11.5, color=INK)
    a1 = ax[0, 1]
    bins = np.logspace(-4, 1.5, 50)
    a1.hist(np.maximum(da16, 1e-4), bins=bins, color=COL["f16"], alpha=.6, label=f"16비트: 중앙값 {np.median(da16):.4f}°, 99% {np.percentile(da16, 99):.3f}°")
    a1.hist(np.maximum(da8, 1e-4), bins=bins, color=COL["u8"], alpha=.6, label=f"8비트: 중앙값 {np.median(da8):.3f}°, 99% {np.percentile(da8, 99):.2f}°, 최대 {da8.max():.2f}°")
    a1.set_xscale("log"); a1.set_xlabel("정책 평균 행동의 차이 (5초 동안 기수 변화, 도)", color=MUTED)
    a1.set_title(f"열린 고리: 같은 관측을 줄였을 때 행동이 얼마나 바뀌나 ({len(R):,}번의 결정)", loc="left", fontsize=11.5, color=INK)
    a1.legend(frameon=False, fontsize=9)
    a2 = ax[1, 0]
    ks = sorted({r["k"] for r in res})
    w = .26
    for j, m in enumerate(MODES):
        v = [next(r["pos_holdout"] for r in res if r["k"] == k and r["mode"] == m) for k in ks]
        a2.bar(np.arange(len(ks)) + (j - 1)*w, v, w - .02, color=COL[m], label=f"{NAME[m]} 평균 {np.mean(v):.3f}")
        for i, x in enumerate(v):
            a2.text(i + (j - 1)*w, x + .01, f"{x:.3f}", ha="center", fontsize=8.5)
    a2.set_xticks(range(len(ks)), [f"지리산 {k + 1}번" for k in ks])
    a2.set_ylabel("별도 particle POS", color=MUTED); a2.set_ylim(0, 1)
    a2.set_title("닫힌 고리: 줄인 관측으로 끝까지 날린 결과 (결정적 정책)", loc="left", fontsize=11.5, color=INK)
    a2.legend(frameon=False, fontsize=9, loc="upper left")
    a3 = ax[1, 1]
    drift = {}
    for m in ("f16", "u8"):
        for k in ks:
            x0 = np.array(next(r["xy"] for r in res if r["k"] == k and r["mode"] == "f32"))
            x1 = np.array(next(r["xy"] for r in res if r["k"] == k and r["mode"] == m))
            n = min(len(x0), len(x1))
            d = np.hypot(*(x1[:n] - x0[:n]).T)/1000
            drift[f"{m} k{k}"] = (float(np.median(d)), float(d.max()), int(np.argmax(d > 0.5)) if (d > 0.5).any() else None)
            a3.plot(np.arange(n)*5/60, d, color=COL[m], lw=1.4, alpha=.8, ls=["-", "--", ":"][k], label=f"{NAME[m]} 지리산 {k + 1}번")
    a3.set_yscale("symlog", linthresh=0.01); a3.set_xlabel("비행 시간 (분)", color=MUTED); a3.set_ylabel("32비트 비행과의 거리 (km)", color=MUTED)
    a3.set_title("닫힌 고리: 비행 경로가 32비트 비행에서 얼마나 벗어나나", loc="left", fontsize=11.5, color=INK)
    a3.legend(frameon=False, fontsize=8.5, ncol=2)
    for a_ in ax.ravel():
        a_.set_facecolor(BG); a_.grid(color=GRID, lw=.5); a_.tick_params(colors=MUTED)
        for sp in a_.spines.values():
            sp.set_color(GRID)
    b = c["bytes"]
    fig.suptitle(f"관측 지도 16·8비트 저장 확인 (R15 800만, 지리산 검증 3개, 학습 없음): 결정 한 번의 지도 {b['f32']/1024:.0f} KB → 16비트 {b['f16']/1024:.0f} KB, "
                 f"8비트 {b['u8']/1024:.0f} KB", x=.01, ha="left", fontsize=13, color=INK)
    fig.tight_layout()
    fig.savefig(OUT/"obs_quant_check.png", dpi=78, facecolor=BG)
    return dict(open_loop=dict(n=len(R), act_deg_f16=dict(median=float(np.median(da16)), p99=float(np.percentile(da16, 99)), max=float(da16.max())),
                               act_deg_u8=dict(median=float(np.median(da8)), p99=float(np.percentile(da8, 99)), max=float(da8.max())),
                               value_f16=dict(median=float(np.median(dv16)), max=float(dv16.max())),
                               value_u8=dict(median=float(np.median(dv8)), max=float(dv8.max())),
                               value_scale=float(np.std(R[:, 3]))),
                closed_loop={f"{r['mode']} k{r['k']}": dict(holdout=r["pos_holdout"], belief=r["pos_belief"]) for r in res},
                drift_km=drift, ranges={kk: dict(min=v[0], max=v[1], over1=v[2]/max(v[3], 1)) for kk, v in rng.items()}, cost=c)


if __name__ == "__main__":
    main()

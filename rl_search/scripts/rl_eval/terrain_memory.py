"""How much memory the terrain, road network and scenario sampler hold, and whether slimming them changes any result (user 2026-11-17:
slim the terrain memory before the 120 km maps, and check the results stay the same).

Run once with the code before a change and once after, then compare:
    python -m rl_search.scripts.rl_eval.terrain_memory measure --label before     # memory per array and per process (fresh process per map)
    python -m rl_search.scripts.rl_eval.terrain_memory hashes --label before      # value fingerprints of every layer and road-grid array
    python -m rl_search.scripts.rl_eval.terrain_memory flights --label before     # R12 flown on three validation cases: POS and observations
    python -m rl_search.scripts.rl_eval.terrain_memory compare --before before --after after

Fingerprints are taken on values (integers as int64, floats as float64, booleans as uint8), so an array stored in a smaller type with
the same values counts as unchanged. Memory per process is the resident size (ps) after loading a map with rl_real.preload, and after
building the RL env and flying 30 decisions of one case.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np

REGIONS = ("amboy", "mojave")
LAYERS = ("z", "water", "fords", "mountain", "forest", "stream", "surface", "city", "built", "forbidden", "slope_deg", "clearance",
          "cover", "cover_gx", "cover_gy", "tpi", "clearance_gx", "clearance_gy", "z_gx", "z_gy")
BASE = ("z", "water", "fords", "mountain", "forest", "stream", "surface", "city", "built")
OUT = Path("rl_search/data/real/amboy/rl_eval/report")
SCRATCH = OUT/"terrain_memory"
MODEL = "rl_search/data/rl_real/r12_2map/ckpt/model_6000000.zip"
CASES = (("bval", 0), ("nval", 1), ("nval", 8))          # Barstow road 1, Amboy mixed 2, Amboy desert 9


def rss_mb() -> float:
    return int(subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())], capture_output=True, text=True).stdout.strip())/1024


def peak_mb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r/2**20 if sys.platform == "darwin" else r/1024


def arrays_of(obj, skip=()) -> dict:
    """Numpy arrays held in the object's attributes (materialised ones only), MB."""
    return {k: v.nbytes/1e6 for k, v in vars(obj).items() if isinstance(v, np.ndarray) and k not in skip and v.nbytes > 1e5}


def fingerprint(a) -> str:
    a = np.asarray(a)
    if a.dtype == bool:
        b = a.astype(np.uint8)
    elif np.issubdtype(a.dtype, np.integer):
        b = a.astype(np.int64)
    else:
        b = a.astype(np.float64)
    return hashlib.sha1(np.ascontiguousarray(b).tobytes()).hexdigest()[:16] + f":{a.shape}"


def one(region: str) -> dict:
    """In a fresh process: memory after loading the map, and after the RL env flew 30 decisions."""
    warnings.filterwarnings("ignore")
    import numpy  # noqa: F401
    r0 = rss_mb()
    from rl_search.rl_real import preload
    st = preload(region)
    t, net, smp = st["terr"], st["net"], st["sampler"]
    r1 = rss_mb()
    lay = {k: v for k, v in arrays_of(t).items()}
    lay_names = [k for k in LAYERS if isinstance(vars(t).get(k), np.ndarray)]
    out = dict(region=region, rss_start=r0, rss_loaded=r1, terrain=lay, terrain_materialised=lay_names,
               net=arrays_of(net, skip=("terr",)), sampler=arrays_of(smp))
    from rl_search.rl_real_eval import _case, _env
    from rl_search.scripts.rl_eval.signal_check import settings_of
    settings = settings_of(json.loads(Path("rl_search/data/rl_real/r12_2map/config.json").read_text()))
    sp, k = ("bval", 0) if region == "mojave" else ("nval", 1)
    case, scen, vres, hold = _case(sp, 3, k)
    env = _env(settings, case.region)
    obs, info = env.reset(options=case.options(scen))
    for _ in range(30):
        obs, r, term, trunc, info = env.step(np.zeros(1, np.float32))
    out.update(rss_env=rss_mb(), peak=peak_mb(), terrain_after_env=[k for k in LAYERS if isinstance(vars(t).get(k), np.ndarray)])
    # the env's map-sized grids: shared by all envs of the process (and forked workers) or one per env
    e = env.env
    grids = dict(visibility=e.detector.A, vis_copy=env.vis_grids[0][1], landcover=e.detector.landcover, passable=e.passable, seen=e.seen)
    shared_ids = set()
    try:
        from rl_search.detection import _LANDCOVER, _VIS
        from rl_search.rl_env import _PASSABLE
        shared_ids = {id(v[2]) for v in _VIS.values()} | {id(v) for v in _LANDCOVER.values()} | {id(v[1]) for v in _PASSABLE.values()}
    except ImportError:
        pass
    out["env_grids"] = {k: dict(mb=v.nbytes/1e6, shared=id(v) in shared_ids, same_as_visibility=(k == "vis_copy" and v is e.detector.A))
                        for k, v in grids.items()}
    return out


def measure(label: str):
    SCRATCH.mkdir(parents=True, exist_ok=True)
    res = {}
    for r in REGIONS:
        p = subprocess.run([sys.executable, "-m", "rl_search.scripts.rl_eval.terrain_memory", "_one", r], capture_output=True, text=True)
        if p.returncode:
            raise RuntimeError(p.stderr[-3000:])
        res[r] = json.loads(p.stdout.strip().splitlines()[-1])
        x = res[r]
        print(f"{r}: process {x['rss_start']:.0f} -> loaded {x['rss_loaded']:.0f} -> RL env {x['rss_env']:.0f} MB (peak {x['peak']:.0f}); "
              f"terrain arrays {sum(x['terrain'].values()):.0f} MB, road grid {sum(x['net'].values()):.0f} MB, sampler {sum(x['sampler'].values()):.0f} MB")
    (SCRATCH/f"measure_{label}.json").write_text(json.dumps(res, indent=1))


def hashes(label: str):
    warnings.filterwarnings("ignore")
    from rl_search.rl_real import preload
    res = {}
    for r in REGIONS:
        st = preload(r)
        t, net = st["terr"], st["net"]
        h = {k: fingerprint(getattr(t, k)) for k in LAYERS}
        h["mountain_z_m"] = repr(t.mountain_z_m)
        for k in ("cell_edge", "cell_s", "road_dist", "near_r", "near_c", "edge_core", "edge_main", "allowed", "out"):
            h["net." + k] = fingerprint(getattr(net, k))
        for lim in (15.0, 30.0):
            net.set_drive_mask(lim)
            h[f"net.drive{lim:g}"] = fingerprint(net.drive)
        h["sampler.perimeter"] = fingerprint(st["sampler"].perimeter)
        h["sampler.lkp_cells"] = fingerprint(st["sampler"].lkp_cells)
        res[r] = h
        print(r, "fingerprints:", len(h))
    SCRATCH.mkdir(parents=True, exist_ok=True)
    (SCRATCH/f"hashes_{label}.json").write_text(json.dumps(res, indent=1))


def flights(label: str):
    warnings.filterwarnings("ignore")
    from rl_search.rl_real_eval import _case, _env, _model
    from rl_search.scripts.rl_eval.signal_check import settings_of
    settings = settings_of(json.loads(Path("rl_search/data/rl_real/r12_2map/config.json").read_text()))
    model = _model(MODEL)
    res = {}
    for sp, k in CASES:
        case, scen, vres, hold = _case(sp, 3, k)
        env = _env(settings, case.region)
        obs, info = env.reset(options=case.options(scen))
        h = hashlib.sha1()
        pos, term, trunc = [float(info["POS"])], False, False
        while not (term or trunc):
            for v in (obs.values() if isinstance(obs, dict) else [obs]):
                h.update(np.ascontiguousarray(v).tobytes())
            a, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(a)
            pos.append(float(info["POS"]))
        res[f"{sp}{k}"] = dict(pos=pos, obs_hash=h.hexdigest(), steps=len(pos) - 1)
        print(f"{sp} {k}: {len(pos) - 1} decisions, final POS {pos[-1]:.4f}")
    SCRATCH.mkdir(parents=True, exist_ok=True)
    (SCRATCH/f"flights_{label}.json").write_text(json.dumps(res))


def compare(before: str, after: str):
    L = [f"# 지형 메모리 줄이기: {before} → {after}", ""]
    ok_all = True
    hb, ha = (json.loads((SCRATCH/f"hashes_{x}.json").read_text()) for x in (before, after))
    for r in REGIONS:
        diff = [k for k in hb[r] if hb[r][k] != ha[r].get(k)]
        ok_all &= not diff
        L.append(f"- {r}: 값 지문 {len(hb[r])}개 중 다른 것 {len(diff)}개" + (f" ({', '.join(diff)})" if diff else ""))
    fb, fa = (json.loads((SCRATCH/f"flights_{x}.json").read_text()) for x in (before, after))
    for c in fb:
        same = fb[c]["pos"] == fa[c]["pos"] and fb[c]["obs_hash"] == fa[c]["obs_hash"]
        ok_all &= same
        L.append(f"- R12 비행 {c}: 결정 {fb[c]['steps']}번, 최종 POS {fb[c]['pos'][-1]:.4f} / {fa[c]['pos'][-1]:.4f}, 매 걸음 POS와 관측 "
                 + ("같음" if same else "다름"))
    mb, ma = (json.loads((SCRATCH/f"measure_{x}.json").read_text()) for x in (before, after))
    L += ["", "| 지도 | 무엇 | 전 (MB) | 후 (MB) |", "|---|---|---|---|"]
    for r in REGIONS:
        b, a = mb[r], ma[r]
        for name, f in (("지형 층 (메모리에 있는 것)", lambda x: sum(x["terrain"].values())), ("도로 격자", lambda x: sum(x["net"].values())),
                        ("출발점 뽑기", lambda x: sum(x["sampler"].values())), ("프로세스: 지도 읽은 뒤", lambda x: x["rss_loaded"] - x["rss_start"]),
                        ("프로세스: RL 환경까지", lambda x: x["rss_env"] - x["rss_start"]), ("프로세스 최고치", lambda x: x["peak"])):
            L.append(f"| {r} | {name} | {f(b):.0f} | {f(a):.0f} |")
        for name, i in (("RL 환경 격자: 함께 씀", 0), ("RL 환경 격자: 환경마다 따로", 1)):
            L.append(f"| {r} | {name} | {env_grids(b, a)[i]:.0f} | {env_grids(a, a)[i]:.0f} |")
    L += ["", "프로세스 메모리(ps)는 맥이 메모리를 압축해 오르내리므로 참고만 함. 배열 크기는 정확함.",
          f"학습 어림(작업자 {WORKERS}개, 두 지도, 배열만): 부모가 만든 것은 fork로 함께 쓰고, 환경 격자 중 따로인 것은 작업자마다 하나."]
    (OUT/f"terrain_memory_{before}_{after}.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    figure(mb, ma, before, after, ok_all)


def env_grids(m, ref):
    """The env grids of a measurement: (shared MB, per-env MB). Measurements made before the grids were recorded (the code before
    2026-11-17) made all of them per env and copied the visibility map for the observation; their sizes are those of ``ref``."""
    g = m.get("env_grids")
    if g is None:
        return 0.0, sum(v["mb"] for v in ref["env_grids"].values())
    shared = sum(v["mb"] for k, v in g.items() if v["shared"] and not v.get("same_as_visibility"))
    per = sum(v["mb"] for k, v in g.items() if not v["shared"] and not v.get("same_as_visibility"))
    return shared, per


WORKERS = 20


def figure(mb, ma, before, after, ok_all):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["AppleGothic", "NanumGothic", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False
    INK, MUTED, GRID, BG = "#1f1f1e", "#6b6b67", "#e4e3de", "#fcfcfb"
    COL = {"base": "#2a78d6", "derived": "#eb6834", "net": "#1baf7a", "sampler": "#7a52c7", "env_shared": "#0b2f5e", "env_per": "#d6336c"}
    NAME = {"base": "지형 기본 층 (파일에서 읽음)", "derived": "지형 계산 층 (기울기·덮개·거리 등)", "net": "도로 격자 (입자용)",
            "sampler": "출발점 뽑기", "env_shared": "RL 환경 격자: 함께 씀 (가시도·땅 덮개·지나갈 칸)", "env_per": "RL 환경 격자: 환경마다 따로"}
    fig, axs = plt.subplots(1, 3, figsize=(23, 7.4), facecolor=BG, gridspec_kw=dict(width_ratios=[1, 1, 1.15]))

    def parts(m, ref):
        sh, per = env_grids(m, ref)
        return dict(base=sum(v for k, v in m["terrain"].items() if k in BASE), derived=sum(v for k, v in m["terrain"].items() if k not in BASE),
                    net=sum(m["net"].values()), sampler=sum(m["sampler"].values()), env_shared=sh, env_per=per)

    for j, r in enumerate(REGIONS):
        ax = axs[j]
        shown = set()
        for i, m in enumerate((mb[r], ma[r])):
            left = 0.0
            for k, v in parts(m, ma[r]).items():
                lab = NAME[k] if v > 0 and k not in shown else None
                shown |= {k} if lab else set()
                ax.barh(i, v, left=left, color=COL[k], height=0.55, label=lab, edgecolor=BG, lw=1.2)
                left += v
            ax.text(left + 8, i, f"{left:.0f} MB", va="center", fontsize=10, color=INK)
        ax.set_yticks([0, 1], ["줄이기 전", "줄인 뒤"], fontsize=10.5, color=INK); ax.invert_yaxis()
        ax.set_xlabel("배열 메모리 (MB): 지도 하나 + RL 환경 하나", fontsize=9, color=MUTED)
        ax.set_title(f"{'암보이' if r == 'amboy' else '바스토'} 60 km: 지도를 읽고 RL 환경 하나를 만들었을 때", loc="left", fontsize=11.5, color=INK)
        h, l = ax.get_legend_handles_labels()
        order = np.argsort([list(NAME.values()).index(x) for x in l])
        ax.legend([h[i] for i in order], [l[i] for i in order], fontsize=8.3, frameon=False, loc="lower right")
        ax.set_xlim(0, 1000)
    # training: env workers share what the parent made before the fork; each worker has its own per-env grids
    ax = axs[2]
    labels, sh_v, per_v = [], [], []
    for scale, sname in ((1, "60 km"), (4, "120 km (어림: 4배)")):
        for tag, m in (("전", mb), ("후", ma)):
            sh = per = 0.0
            for r in REGIONS:
                p_ = parts(m[r], ma[r])
                sh += (p_["base"] + p_["derived"] + p_["net"] + p_["sampler"] + p_["env_shared"])*scale
                per += p_["env_per"]*scale*WORKERS
            labels.append(f"{sname}\n줄이기 {tag}"); sh_v.append(sh/1000); per_v.append(per/1000)
    x0 = np.arange(len(labels))
    ax.bar(x0, sh_v, color="#2a78d6", width=0.62, label="부모가 만들어 작업자들이 함께 씀 (지형, 도로 격자, 공유 격자)")
    ax.bar(x0, per_v, bottom=sh_v, color="#d6336c", width=0.62, label=f"작업자마다 따로 × {WORKERS} (환경 격자)")
    for xx, a_, b_ in zip(x0, sh_v, per_v):
        ax.text(xx, a_ + b_ + 0.08, f"{a_ + b_:.1f} GB", ha="center", fontsize=10, color=INK)
    ax.set_xticks(x0, labels, fontsize=9.5, color=INK)
    ax.set_ylabel("GB (두 지도, 배열만)", fontsize=9, color=MUTED)
    ax.set_title(f"학습할 때: 환경 작업자 {WORKERS}개, 두 지도 (배열만 센 어림)", loc="left", fontsize=11.5, color=INK)
    ax.legend(fontsize=8.5, frameon=False, loc="upper left")
    for a in axs:
        a.grid(axis="x" if a is not axs[2] else "y", color=GRID, lw=.6); a.set_facecolor(BG); a.tick_params(colors=MUTED, labelsize=8.5)
        for s in a.spines.values():
            s.set_color(GRID)
    fig.suptitle("지형 메모리 줄이기 (60 km 지도 두 장). 결과: " + ("층 값, 도로 격자, 입자, R12 비행이 줄이기 전과 모두 같음" if ok_all else "다른 것이 있음 (표 참고)"),
                 x=0.01, ha="left", fontsize=13.5, color=INK)
    fig.tight_layout()
    fig.savefig(OUT/f"terrain_memory_{before}_{after}.png", dpi=90, facecolor=BG)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=("measure", "hashes", "flights", "compare", "_one"))
    ap.add_argument("region", nargs="?")
    ap.add_argument("--label", default="now")
    ap.add_argument("--before", default="before")
    ap.add_argument("--after", default="after")
    a = ap.parse_args()
    if a.what == "_one":
        print(json.dumps(one(a.region)))
    elif a.what == "measure":
        measure(a.label)
    elif a.what == "hashes":
        hashes(a.label)
    elif a.what == "flights":
        flights(a.label)
    else:
        compare(a.before, a.after)


if __name__ == "__main__":
    main()

"""How many worlds the local PC should run for R18 (user 2026-11-24: "환경수는 코어수와 램에 맞춰서"): ``r18_start.sh`` is started for a few
updates at each SCALE (SCALE x (8, 6, 5, 4) worlds of 1 / 2 / 3 / 4 UAVs), with no evaluations, into a scratch folder; per SCALE the
table gets the decisions per second of the last updates (collection and PPO update together), seconds per update and the lowest free
memory (``MemAvailable``) seen. The run with the most decisions per second that keeps at least ``--mem-margin`` of the memory free is
proposed. Linux (``/proc/meminfo``); run from the repository root on the local PC.

    python -m rl_search.scripts.rl_eval.scale_test --scales 2 4 6 8 10 11 --updates 4 [--w 1] [--gpu 0]
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

LOGDIR = Path("rl_search/data/rl_real")


def mem_available_gb() -> tuple:
    info = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    kb = lambda key: float(info[key].strip().split()[0])
    return kb("MemAvailable")/2**20, kb("MemTotal")/2**20


def one(scale: int, updates: int, w: int, gpu: str, scratch: Path) -> dict:
    tag = f"r18_scale_s{scale}"
    log = LOGDIR/f"{tag}_train.log"
    log.unlink(missing_ok=True)
    env = dict(os.environ, TAG=tag, SCALE=str(scale), W=str(w), GPU=gpu, STOP=str(updates),
               EXTRA=f"--out {scratch} --eval-workers 1 --ring-eval-every 1000 --eval-every 1000 --save-every 1000")
    t0 = time.time()
    p = subprocess.Popen(["bash", "rl_search/scripts/rl_eval/r18_start.sh"], env=env)
    low, total = mem_available_gb()
    while p.poll() is None:
        time.sleep(5)
        low = min(low, mem_available_gb()[0])
    text = log.read_text() if log.exists() else ""
    fps = [int(x) for x in re.findall(r"\|\s+fps\s+\|\s+(\d+)", text)]
    steps = [int(x) for x in re.findall(r"\|\s+total_timesteps\s+\|\s+(\d+)", text)]
    elapsed = [float(x) for x in re.findall(r"\|\s+time_elapsed\s+\|\s+(\d+)", text)]
    m = re.search(r"multi-UAV: worlds \[.*\], (\d+) UAV slots", text)
    slots = int(m.group(1)) if m else 0
    per_update = (elapsed[-1] - elapsed[-2]) if len(elapsed) >= 2 else float("nan")
    rate = (steps[-1] - steps[-2])/per_update if len(steps) >= 2 and per_update > 0 else (fps[-1] if fps else 0)
    ok = "stopped at" in text
    shutil.rmtree(scratch/tag, ignore_errors=True)
    return dict(scale=scale, worlds=23*scale, slots=slots, rate=rate, sec_update=per_update, mem_low=low, mem_total=total,
                minutes=(time.time() - t0)/60, ok=ok, log=str(log))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scales", type=int, nargs="+", default=[2, 4, 6, 8, 10, 11])
    ap.add_argument("--updates", type=int, default=4)
    ap.add_argument("--w", type=int, default=1, help="worlds per process")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--mem-margin", type=float, default=0.15, help="keep at least this share of the memory free")
    ap.add_argument("--scratch", default="/tmp/r18_scale")
    a = ap.parse_args()
    scratch = Path(a.scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    rows = []
    for s in a.scales:
        r = one(s, a.updates, a.w, a.gpu, scratch)
        rows.append(r)
        print(f"SCALE {s:2d}: {r['worlds']} worlds, {r['slots']} UAV slots | {r['rate']:7.0f} decisions/s (last update) | "
              f"{r['sec_update']:5.0f} s per update | lowest free memory {r['mem_low']:.0f} of {r['mem_total']:.0f} GB | "
              f"{r['minutes']:.1f} min | {'ok' if r['ok'] else 'FAILED, see ' + r['log']}", flush=True)
        if r["mem_low"] < a.mem_margin*r["mem_total"]:
            print(f"  free memory below {a.mem_margin:.0%}: larger SCALEs not tried", flush=True)
            break
    good = [r for r in rows if r["ok"] and r["mem_low"] >= a.mem_margin*r["mem_total"]]
    if good:
        best = max(good, key=lambda r: r["rate"])
        print(f"proposed: SCALE {best['scale']} ({best['worlds']} worlds, {best['slots']} slots, {best['rate']:.0f} decisions/s, "
              f"{best['sec_update']:.0f} s per update; 2,000 updates about {2000*best['sec_update']/3600:.1f} h)")


if __name__ == "__main__":
    main()

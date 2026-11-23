"""Evaluate a multi-UAV run's checkpoints as they appear (user 2026-11-23, M2): every new ``<run>/ckpt/model_<step>.zip`` is flown by
``ma_eval run`` (Jirisan v2j, 1-4 UAVs, web runs saved) with the label ``<prefix>_<step / 1e6 rounded to 0.1>M``, one at a time, oldest first;
stops when the training process is gone and every checkpoint is done. A baseline (``--baseline-model``, flown with ``--single``) goes first.

    nohup .venv/bin/python -m rl_search.scripts.multi_design.ma_eval_watch --run rl_search/data/rl_real/m2_multi --prefix m2 \
        --baseline-model rl_search/data/rl_real/r15_chungju/ckpt/model_6800000.zip --baseline-label r15_6.8M_m2 --design m2 > log 2>&1 &
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

from rl_search.scripts.multi_design.ma_eval import WORK


def training_alive(tag: str) -> bool:
    out = subprocess.run(["ps", "-ax", "-o", "command="], capture_output=True, text=True).stdout
    return any("train_rl" in line and f"--tag {tag}" in line for line in out.splitlines())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run folder (with ckpt/ and config.json)")
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--design", default=None, help="env design for the baseline (the run's checkpoints use their config's)")
    ap.add_argument("--baseline-model", default=None)
    ap.add_argument("--baseline-label", default=None)
    ap.add_argument("--poll", type=float, default=60.0)
    ap.add_argument("--every", type=int, default=0, help="only checkpoints at multiples of this many steps (0 = every checkpoint)")
    ap.add_argument("--teams", type=int, nargs="+", default=None, help="team sizes per checkpoint (default 1 2 3 4)")
    ap.add_argument("--no-holdout", action="store_true", help="planner score only (no hold-out replay) per checkpoint")
    ap.add_argument("--final-full", action="store_true", help="when training ends: the last checkpoint with every team size and the hold-out replay")
    a = ap.parse_args()
    run = Path(a.run)
    tag = run.name
    py = sys.executable

    def ma_eval(*args):
        cmd = [py, "-m", "rl_search.scripts.multi_design.ma_eval", "run", *args]
        print(time.strftime("%m-%d %H:%M"), " ".join(cmd[3:]), flush=True)
        subprocess.run(["nice", "-n", "15", *cmd], check=False)

    if a.baseline_model and a.baseline_label and not (WORK/f"{a.baseline_label}.json").exists():
        ma_eval("--model", a.baseline_model, "--label", a.baseline_label, "--single", *(["--design", a.design] if a.design else []))
    done = set()
    while True:
        ck = sorted(run.glob("ckpt/model_*.zip"), key=lambda p: int(re.search(r"(\d+)", p.stem).group(1)))
        todo = [p for p in ck if p not in done]
        for p in todo:
            step = int(re.search(r"(\d+)", p.stem).group(1))
            label = f"{a.prefix}_{round(step/1e5)/10:g}M"
            on_grid = not a.every or abs(step - round(step/a.every)*a.every) < 20000
            if on_grid and "saturated" not in p.stem and not (WORK/f"{label}.json").exists():
                time.sleep(20)                                   # the checkpoint file may still be being written
                ma_eval("--model", str(p), "--label", label, *(["--teams", *map(str, a.teams)] if a.teams else []),
                        *(["--no-holdout"] if a.no_holdout else []))
            done.add(p)
        if not todo and not training_alive(tag):
            if a.final_full and ck:
                last = ck[-1]
                step = int(re.search(r"(\d+)", last.stem).group(1))
                label = f"{a.prefix}_{round(step/1e5)/10:g}M_full"
                if not (WORK/f"{label}.json").exists():
                    ma_eval("--model", str(last), "--label", label)
            print(time.strftime("%m-%d %H:%M"), "training gone, all checkpoints evaluated", flush=True)
            break
        time.sleep(a.poll)


if __name__ == "__main__":
    main()

"""An RL checkpoint on the validation or test cases, scored like the SAROPS references (rl_search.rl_real_eval), with a table against
SAROPS when its references exist (scripts/rl_eval/sarops_ref.py). Flights are saved as web runs (``eval-<split>-s<stage>-k<k>-rl-<label>``).

    python -m rl_search.scripts.rl_eval.eval_rl --model rl_search/data/rl_real/amboy_v1/best_stage3.zip --label 1000000 --split test --stages 3
Results: <model folder>/eval_<split>_<label>.json
"""
from __future__ import annotations

import argparse
import json
from multiprocessing import get_context
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--label", required=True, help="step or name shown in the web run id")
    ap.add_argument("--split", default="test", choices=("val", "test"))
    ap.add_argument("--stages", type=int, nargs="+", default=[3])
    ap.add_argument("--cases", type=int, default=None, help="per stage (default 5 for val, 30 for test)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--first", type=int, default=0, help="first case index (for a quick check)")
    args = ap.parse_args()
    from rl_search.rl_real import preload
    from rl_search.rl_real_eval import eval_job, sarops_refs
    preload("amboy")
    n = args.cases or (5 if args.split == "val" else 30)
    label = int(args.label) if args.label.isdigit() else args.label
    jobs = [(str(args.model), args.split, st, k, label) for st in args.stages for k in range(args.first, n)]
    with get_context("fork").Pool(args.workers) as pool:
        rows = pool.map(eval_job, jobs, chunksize=1)
    try:
        refs = sarops_refs(args.split)
    except FileNotFoundError:
        refs = {}
    for r in rows:
        ref = refs.get((r["stage"], r["k"]))
        r["sarops"] = ref["holdout"] if ref else None
    out = args.model.parent/f"eval_{args.split}_{args.label}.json"
    out.write_text(json.dumps(rows, indent=1))
    have = [r for r in rows if r["sarops"]]
    if have:
        rl, sa = np.array([r["holdout"]["total"] for r in have]), np.array([r["sarops"]["total"] for r in have])
        d = rl - sa
        print(f"{len(have)} cases: RL {rl.mean():.3f}, SAROPS {sa.mean():.3f}, paired difference {d.mean():+.3f} ± {d.std(ddof=1)/np.sqrt(len(d)):.3f} "
              f"(standard error), RL higher in {(d > 0).sum()}/{len(d)}")
    print("saved", out)


if __name__ == "__main__":
    main()

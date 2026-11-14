"""An RL checkpoint on the validation or test cases, scored like the SAROPS references (rl_search.rl_real_eval), with a table against
SAROPS when its references exist (scripts/rl_eval/sarops_ref.py). Flights are saved as web runs (``eval-<split>-s<stage>-k<k>-rl-<label>``).

    python -m rl_search.scripts.rl_eval.eval_rl --model rl_search/data/rl_real/amboy_v1/best_stage3.zip --label 1000000 --split test --stages 3 \
        --area-def window
Results: <model folder>/eval_<split>_<label>.json. ``--area-def`` window (x 70, default, the amboy_v1 A) or seen (the seen-particle A
option of 2026-11-11, x 80, scored against sarops_ref_<split>_seen.json).
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
    ap.add_argument("--split", default="test", choices=("val", "test", "dval", "dtest", "mval", "mtest", "nval", "ntest"))
    ap.add_argument("--stages", type=int, nargs="+", default=[3])
    ap.add_argument("--cases", type=int, default=None, help="per stage (default 5 for val, 30 for test)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--first", type=int, default=0, help="first case index (for a quick check)")
    ap.add_argument("--area-def", default="window", choices=("window", "realtime", "realtime_q", "realtime_t", "seen"))
    ap.add_argument("--blur-tau", type=float, default=0.0)
    ap.add_argument("--area-roll-max", type=float, default=20.0)
    ap.add_argument("--pos-w", type=float, default=1.0)
    ap.add_argument("--blur-dead-roll", type=float, default=0.0, help="with --blur-tau: no turn-rate loss up to this roll's turn rate (D3)")
    ap.add_argument("--blur-tau-low", type=float, default=0.0, help="with --blur-dead-roll: exp(-w/value) below that rate too (D4 = D2 x D3)")
    ap.add_argument("--area-sigma", type=float, default=0.0, help="blur width (m) of the real-time A region (0 = 150 m)")
    ap.add_argument("--area-sigma-off", type=float, default=0.0, help="blur width (m) of the off-road particles in the A region (0 = same)")
    ap.add_argument("--handover-km", type=float, default=0.0, help="distance hand-over of the model's training (0 = 2 min after the launch)")
    ap.add_argument("--no-edge", action="store_true", help="the model was trained without the edge mask (maps show 0 off the map)")
    ap.add_argument("--area-img-min", type=float, default=0.0, help="a cell is searched only where the POD's image-position factor is >= this")
    ap.add_argument("--big-map-km", type=float, default=0.0, help="side (km) of the model's largest observation map (0 = 51.2)")
    ap.add_argument("--rel-maps", action="store_true", help="the model's maps have the relative-POS channel (train_rl --rel-maps)")
    ap.add_argument("--a-weight", type=float, default=1.0, help="the model's A weight (0 = A off, scored on POS only)")
    ap.add_argument("--pos-norm", action="store_true", help="the model's maps use per-map POS normalisation (train_rl --pos-norm)")
    ap.add_argument("--fp-rate", action="store_true", help="the model's camera channel carries the turn-rate factor (train_rl --fp-rate)")
    ap.add_argument("--no-area", action="store_true", help="the model was trained without A in the env (train_rl --no-area)")
    ap.add_argument("--obs-r10", action="store_true", help="the model's maps are the R10 channels (train_rl --obs-r10)")
    ap.add_argument("--ks", type=int, nargs="+", default=None, help="only these case indices (0-based), e.g. 2 5 8 11 14 = the desert mval cases")
    ap.add_argument("--launch-min", type=float, default=None,
                    help="launch this many minutes after the report instead of each case's own delay (same particles and start point); "
                         "no SAROPS comparison. Put it in --label too, the web run id is eval-<split>-s3-k<k>-rl-<label>")
    args = ap.parse_args()
    from rl_search.scripts.rl_eval.sarops_ref import DEFS
    over = ({} if args.area_def == "window" else dict(DEFS[args.area_def])) | ({"blur_tau": args.blur_tau} if args.blur_tau else {}) \
        | ({"area_roll_max": args.area_roll_max} if args.area_def == "realtime_t" else {}) | ({"area_w": args.pos_w} if args.pos_w != 1 else {}) \
        | ({"blur_dead_roll": args.blur_dead_roll} if args.blur_tau and args.blur_dead_roll else {}) \
        | ({"blur_tau_low": args.blur_tau_low} if args.blur_tau and args.blur_dead_roll and args.blur_tau_low else {}) \
        | ({"area_sigma": args.area_sigma} if args.area_sigma and args.area_sigma != 150 else {}) \
        | ({"area_sigma_off": args.area_sigma_off} if args.area_sigma_off else {}) \
        | ({"handover_km": args.handover_km} if args.handover_km else {}) | ({"edge_mode": "none", "off_map_zero": True} if args.no_edge else {}) \
        | ({"area_img_min": args.area_img_min} if args.area_img_min else {}) | ({"big_map_m": args.big_map_km*1000.0} if args.big_map_km else {}) \
        | ({"rel_maps": True} if args.rel_maps else {}) | ({"fp_rate": True} if args.fp_rate else {}) \
        | ({"a_weight": args.a_weight} if args.a_weight != 1.0 else {}) | ({"pos_norm": True} if args.pos_norm else {}) \
        | ({"no_area": True} if args.no_area else {}) | ({"obs_r10": True} if args.obs_r10 else {})
    from rl_search.rl_real import preload
    from rl_search.rl_real_eval import eval_job, sarops_refs
    preload("amboy")
    from rl_search.rl_eval_set import MULTI_N
    n = args.cases or MULTI_N.get(args.split) or (5 if args.split in ("val", "dval") else 30)
    label = int(args.label) if args.label.isdigit() else args.label
    ks = args.ks if args.ks is not None else list(range(args.first, n))
    launch_s = None if args.launch_min is None else 60.0*args.launch_min
    jobs = [(str(args.model), args.split, st, k, label, True, over, launch_s) for st in args.stages for k in ks]
    with get_context("fork").Pool(args.workers) as pool:
        rows = pool.map(eval_job, jobs, chunksize=1)
    try:
        from rl_search.scripts.rl_eval.sarops_ref import ref_key_of
        refs = sarops_refs(args.split, ref_key_of(over | {"area_def": args.area_def})) if launch_s is None else {}
    except FileNotFoundError:
        refs = {}
    if launch_s is not None:
        for r in rows:
            print(f"case {r['k'] + 1}: launch {args.launch_min:g} min, hold-out POS {r['holdout']['pos']:.3f}, {r['sec']:.0f} s", flush=True)
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
    ends = [r["end"] for r in rows if r.get("end")]
    if ends:
        print(f"end of flight (mean of {len(ends)}): POS gained in the last 15 min {np.mean([x['pos_last15'] for x in ends]):.3f}, "
              f"minutes far from the remaining belief (<5 % within 10 km) {np.mean([x['far_min'] for x in ends]):.1f}, "
              f"minutes within 1 km of the map edge {np.mean([x['edge_min'] for x in ends]):.1f}")
    print("saved", out)


if __name__ == "__main__":
    main()

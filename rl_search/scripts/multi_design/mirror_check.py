"""Left-right symmetry of a policy (user 2026-11-23: M2 / M2b keep falling into one-sided turning; "확인해보고 좌우 대칭이 필요한 이유와 그
결과 자세히 알려줘"). The search problem is mirror-symmetric: reflect the world across the UAV's heading line and the right turn becomes the
left turn. So a policy should satisfy mean(mirror(o)) = -mean(o). Every mean action splits into

    symmetric part  s(o) = (mean(o) - mean(mirror(o))) / 2      (turns that follow the situation; flips with the mirror)
    one-sided part  b(o) = (mean(o) + mean(mirror(o))) / 2      (the same turn whatever the side: 0 for a symmetric policy)

``mirror_obs``: every map (heading up, UAV in the centre, columns = right) flipped left-right; in the vector, roll, turn-rate state and course
minus heading change sign and the allowed range becomes (-high, -low); energy, time, belief left, map scales and 1 / team size stay.

    python -m rl_search.scripts.multi_design.mirror_check fly --model <ckpt> --label <name> [--single] --k 4 --case 0 --mode raw|sym
    python -m rl_search.scripts.multi_design.mirror_check figure
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
import warnings
from pathlib import Path

import numpy as np

from rl_search.scripts.multi_design.ma_eval import OUT, RUN_CFG, _gaps

WORK = OUT/"mirror_check"
FLIP_SIGN = (0, 1, 2)                 # roll, turn_rate_state, course_minus_heading (rl_real.VEC_NAMES)
LOW, HIGH = 6, 7                      # allowed_low, allowed_high


def mirror_obs(obs: dict) -> dict:
    """The observation of the mirrored world (see the module doc); works on single observations and batches, float or uint8 maps."""
    out = {}
    for key, v in obs.items():
        if key == "vec":
            m = np.array(v, copy=True)
            m[..., list(FLIP_SIGN)] = -m[..., list(FLIP_SIGN)]
            m[..., LOW], m[..., HIGH] = -v[..., HIGH], -v[..., LOW]
            out[key] = m
        else:
            out[key] = np.ascontiguousarray(v[..., ::-1])       # columns = right: flip left-right
    return out


def means(model, batch: dict) -> np.ndarray:
    import torch
    with torch.no_grad():
        ot, _ = model.policy.obs_to_tensor(batch)
        return model.policy.get_distribution(ot).distribution.mean.cpu().numpy().ravel()


def fly(model_path: str, label: str, single: bool, k: int, case_k: int, mode: str) -> dict:
    """One deterministic team flight on a Jirisan case in the m2 env; ``mode`` "raw" flies the policy's mean, "sym" its symmetric part
    s(o). Records per active decision the mean, the mirrored mean, and per UAV the heading (circling)."""
    warnings.filterwarnings("ignore")
    import torch
    from stable_baselines3 import PPO
    from rl_search.rl_multi import MultiRealEnv
    from rl_search.rl_real import RealRLConfig
    from rl_search.rl_real_eval import _case
    from rl_search.scripts.rl_eval.signal_check import settings_of
    torch.set_num_threads(1)
    c_path = Path(model_path).parent.parent/"config.json"
    c = json.loads((c_path if c_path.exists() else Path(RUN_CFG)).read_text())
    s = settings_of(c)
    if single:
        s = {key: v for key, v in s.items() if key != "obs_u8"}
    model = PPO.load(model_path, device="cpu")
    case, scen, _, _ = _case("v2j", 3, case_k)
    env = MultiRealEnv(k, RealRLConfig(region="jirisan120", settings=s, mask_cost=0.1), seed=case_k, design="m2")
    obs = env.reset(options=case.options(scen), gaps=_gaps(case_k, k, "m2"))
    mu_all, mm_all, heads = [], [], [[] for _ in range(k)]
    while not env.done:
        if single:
            obs = [env.base_observation(i) for i in range(k)]
        batch = {key: np.stack([o[key] for o in obs]) for key in obs[0]}
        mu = means(model, batch)
        mm = means(model, mirror_obs(batch))
        act = env.active.copy()
        mu_all += list(mu[act])
        mm_all += list(mm[act])
        a = np.clip((mu - mm)/2 if mode == "sym" else mu, -1.0, 1.0)
        obs, _, _, info = env.step([np.atleast_1d(x) for x in a])
        for i in range(k):
            if act[i]:
                heads[i].append(float(env.uavs[i].env.plant.pose[2]))
    circ = []
    for h in heads:
        hd = np.unwrap(np.array(h))
        mins = np.arange(0, len(hd) - 36, 12)
        circ.append(float(np.mean([abs(hd[q + 36] - hd[q]) >= np.radians(330) for q in mins])) if len(mins) else np.nan)
    mu_all, mm_all = np.array(mu_all), np.array(mm_all)
    out = dict(label=label, mode=mode, k=k, case=case_k, found=float(env.found), circling=float(np.nanmean(circ)),
               mean=mu_all.tolist(), mirror=mm_all.tolist())
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK/f"{label}_{mode}_k{k}_c{case_k}.json").write_text(json.dumps(out))
    b = (mu_all + mm_all)/2
    print(f"{label:10s} {mode:3s} k={k} case {case_k + 1}: team POS {env.found:.3f}, circling {np.nanmean(circ)*100:3.0f}% | mean {mu_all.mean():+.3f} "
          f"| one-sided part: mean {b.mean():+.3f}, mean |.| {np.abs(b).mean():.3f} | symmetric part mean |.| {np.abs((mu_all - mm_all)/2).mean():.3f}",
          flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("fly", "figure"))
    ap.add_argument("--model")
    ap.add_argument("--label")
    ap.add_argument("--single", action="store_true")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--case", type=int, default=0)
    ap.add_argument("--mode", choices=("raw", "sym"), default="raw")
    a = ap.parse_args()
    if a.what == "fly":
        fly(a.model, a.label, a.single, a.k, a.case, a.mode)


if __name__ == "__main__":
    main()

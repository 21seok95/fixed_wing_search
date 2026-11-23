"""Checks of the left-right symmetric policy (rl_search/sym_policy.py; user 2026-11-23 "①~③ 시험"):

① exact: on real observations (flights on Jirisan cases, R15's float inputs and M2's 8-bit inputs) -- mirroring twice gives the
   observation back; mean(mirror(o)) = -mean(o) and value(mirror(o)) = value(o) to float precision; log p(a | o) = log p(-a | mirror(o));
   the symmetric mean equals (raw mean(o) - raw mean(mirror(o))) / 2 of the same weights (what mirror_check flew); a vector the mirror rule
   does not know is refused.
② reproduce: the symmetric policy flies the Jirisan cases (4 UAVs, deterministic) and must give the flights of mirror_check's "sym" mode
   (team POS of the stored runs).
③ smoke training: see the message log (train_rl --sym-policy); then ``exact`` again on its checkpoint.

    python -m rl_search.scripts.multi_design.sym_policy_check exact --model <ckpt> [--single]
    (a model trained with --sym-policy, e.g. R16 on the Ubuntu PC: its first checkpoint, every line PASS)
    python -m rl_search.scripts.multi_design.sym_policy_check reproduce --model <ckpt> --label R15 [--single] --case 0
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
import warnings
from pathlib import Path

import numpy as np

from rl_search.scripts.multi_design.ma_eval import RUN_CFG, _gaps

MIRROR_RUNS = Path("rl_search/data/real/amboy/rl_eval/report/mirror_check")


def _settings(model_path: str, single: bool) -> dict:
    from rl_search.scripts.rl_eval.signal_check import settings_of
    c_path = Path(model_path).parent.parent/"config.json"
    c = json.loads((c_path if c_path.exists() and not single else Path(RUN_CFG)).read_text())
    s = settings_of(c)
    if single:
        s = {k: v for k, v in s.items() if k != "obs_u8"}
    return s


def _load(model_path: str, sym: bool | None):
    """sym True: the weights in the symmetric policy; False: in the plain network (also for a model trained with --sym-policy, 11-23
    R16); None: as stored."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.policies import MultiInputActorCriticPolicy
    from rl_search.sym_policy import SymmetricMultiInputPolicy
    if sym is None:
        return PPO.load(model_path, device="cpu")
    return PPO.load(model_path, device="cpu", custom_objects={"policy_class": SymmetricMultiInputPolicy if sym else MultiInputActorCriticPolicy})


def _collect(model_path: str, single: bool, n_steps: int = 260, cases=(0, 2)) -> dict:
    """Real observations: the raw policy flies 4 UAVs on Jirisan cases for n_steps decisions; every active UAV's observation is kept."""
    from rl_search.rl_multi import MultiRealEnv
    from rl_search.rl_real import RealRLConfig
    from rl_search.rl_real_eval import _case
    s = _settings(model_path, single)
    model = _load(model_path, sym=None)                  # flies as trained (a --sym-policy model: symmetric)
    keep = []
    for case_k in cases:
        case, scen, _, _ = _case("v2j", 3, case_k)
        env = MultiRealEnv(4, RealRLConfig(region="jirisan120", settings=s, mask_cost=0.1), seed=case_k, design="m2")
        obs = env.reset(options=case.options(scen), gaps=_gaps(case_k, 4, "m2"))
        for _ in range(n_steps):
            if single:
                obs = [env.base_observation(i) for i in range(4)]
            batch = {k: np.stack([o[k] for o in obs]) for k in obs[0]}
            a, _ = model.predict(batch, deterministic=True)
            keep += [{k: v[i] for k, v in batch.items()} for i in np.flatnonzero(env.active)]
            obs, _, _, _ = env.step([np.atleast_1d(x) for x in np.ravel(a)])
    return {k: np.stack([o[k] for o in keep]) for k in keep[0]}


def exact(model_path: str, single: bool):
    warnings.filterwarnings("ignore")
    import torch
    from rl_search.sym_policy import SymmetricMultiInputPolicy, check_layout, mirror_np, mirror_th
    torch.set_num_threads(1)
    O = _collect(model_path, single)
    n = len(O["vec"])
    print(f"{model_path}: {n} real observations, vector {O['vec'].shape[1]}, maps " +
          ", ".join(f"{k} {v.shape[1]}x{v.shape[2]}x{v.shape[3]} {v.dtype}" for k, v in O.items() if k != "vec"))
    mm = mirror_np(mirror_np(O))
    twice = max(float(np.abs(mm[k].astype(float) - O[k].astype(float)).max()) for k in O)
    tt = {k: torch.as_tensor(v) for k, v in O.items()}
    t2 = mirror_th(mirror_th(tt))
    twice_t = max(float((t2[k].double() - tt[k].double()).abs().max()) for k in O)
    th_np = max(float(np.abs(mirror_th(tt)[k].numpy().astype(float) - mirror_np(O)[k].astype(float)).max()) for k in O)
    sym, raw = _load(model_path, sym=True), _load(model_path, sym=False)
    assert isinstance(sym.policy, SymmetricMultiInputPolicy)
    M = mirror_np(O)

    def stats(model, obs):
        mus, vs = [], []
        with torch.no_grad():
            for s in range(0, n, 256):
                ot, _ = model.policy.obs_to_tensor({k: v[s:s + 256] for k, v in obs.items()})
                mus.append(model.policy.get_distribution(ot).distribution.mean.numpy().ravel())
                vs.append(model.policy.predict_values(ot).numpy().ravel())
        return np.concatenate(mus), np.concatenate(vs)

    mu, v = stats(sym, O)
    mu_m, v_m = stats(sym, M)
    r_mu, _ = stats(raw, O)
    r_mm, _ = stats(raw, M)
    a = np.random.default_rng(0).uniform(-1, 1, (n, 1)).astype(np.float32)
    with torch.no_grad():
        ot, _ = sym.policy.obs_to_tensor(O)
        om, _ = sym.policy.obs_to_tensor(M)
        _, lp, _ = sym.policy.evaluate_actions(ot, torch.as_tensor(a))
        _, lpm, _ = sym.policy.evaluate_actions(om, torch.as_tensor(-a))
    refused = False
    from rl_search.rl_real import VEC_NAMES
    from rl_search.sym_policy import EXTRA_KEEP
    try:                                              # one entry more than the mirror rule knows (11-23: EXTRA_KEEP 2 with the turn entry)
        check_layout(len(VEC_NAMES) + len(O) - 1 + EXTRA_KEEP + 1, len(O) - 1)
    except ValueError:
        refused = True
    rows = [("mirror twice = the observation (numpy / torch)", f"{twice:.1e} / {twice_t:.1e}", twice == 0 and twice_t == 0),
            ("torch mirror = numpy mirror", f"{th_np:.1e}", th_np == 0),
            ("mean(o) + mean(mirror(o)), largest", f"{np.abs(mu + mu_m).max():.1e}", np.abs(mu + mu_m).max() < 1e-5),
            ("value(o) - value(mirror(o)), largest (values up to %.0f)" % np.abs(v).max(), f"{np.abs(v - v_m).max():.1e}",
             np.abs(v - v_m).max() < 1e-4*max(1.0, np.abs(v).max())),
            ("log p(a|o) - log p(-a|mirror(o)), largest", f"{float((lp - lpm).abs().max()):.1e}", float((lp - lpm).abs().max()) < 1e-4),
            ("symmetric mean - (raw(o) - raw(mirror(o)))/2, largest", f"{np.abs(mu - (r_mu - r_mm)/2).max():.1e}", np.abs(mu - (r_mu - r_mm)/2).max() < 1e-5),
            ("a vector of unknown length is refused", str(refused), refused)]
    for name, val, ok in rows:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}: {val}")
    print(f"  (for scale: raw mean |m(o)| {np.abs(r_mu).mean():.3f}, raw one-sided part |(m(o)+m(mirror(o)))/2| {np.abs((r_mu + r_mm)/2).mean():.3f}, "
          f"symmetric mean |.| {np.abs(mu).mean():.3f})")
    return all(ok for _, _, ok in rows)


def reproduce(model_path: str, label: str, single: bool, case_k: int):
    """Fly the symmetric policy (deterministic) on a Jirisan case with 4 UAVs and compare with mirror_check's stored "sym" flight."""
    warnings.filterwarnings("ignore")
    import torch
    from rl_search.rl_multi import MultiRealEnv
    from rl_search.rl_real import RealRLConfig
    from rl_search.rl_real_eval import _case
    torch.set_num_threads(1)
    s = _settings(model_path, single)
    model = _load(model_path, sym=True)
    case, scen, _, _ = _case("v2j", 3, case_k)
    env = MultiRealEnv(4, RealRLConfig(region="jirisan120", settings=s, mask_cost=0.1), seed=case_k, design="m2")
    obs = env.reset(options=case.options(scen), gaps=_gaps(case_k, 4, "m2"))
    while not env.done:
        if single:
            obs = [env.base_observation(i) for i in range(4)]
        batch = {k: np.stack([o[k] for o in obs]) for k in obs[0]}
        a, _ = model.predict(batch, deterministic=True)
        obs, _, _, _ = env.step([np.atleast_1d(x) for x in np.ravel(a)])
    ref = json.loads((MIRROR_RUNS/f"{label}_sym_k4_c{case_k}.json").read_text())
    same = abs(env.found - ref["found"]) < 5e-4
    print(f"  {'PASS' if same else 'DIFF'}  {label} case {case_k + 1}: symmetric policy team POS {env.found:.4f} | mirror_check sym {ref['found']:.4f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("exact", "reproduce"))
    ap.add_argument("--model", required=True)
    ap.add_argument("--single", action="store_true")
    ap.add_argument("--label", default=None)
    ap.add_argument("--case", type=int, default=0)
    a = ap.parse_args()
    if a.what == "exact":
        exact(a.model, a.single)
    else:
        reproduce(a.model, a.label, a.single, a.case)


if __name__ == "__main__":
    main()

"""Multi-UAV MDP check (user 2026-11-20, during M1: "상태, 행동, 보상 꼼꼼히 살펴보자"): one team flight on a Jirisan validation case with
a model, every step and UAV recorded, to look at what each UAV is actually rewarded for and what the trainer makes of it.

- reward split per UAV: the team's found POS (its own share and the rest = teammates' finds), the look reward A' (particles in the image
  centre per pass, unweighted, own passes only), the filter / edge cuts, the safety penalty
- A' against what the UAV found itself (own share, own seen): does a UAV flying behind a teammate still collect A'?
- the critic around each UAV's battery end: V(o) of its last active steps and of the steps after it ended (masked, not terminal for SB3),
  and the advantages as the trainer computes them (GAE over the whole slot sequence, the slot's flight end not a terminal) against GAE with
  the flight end as a terminal
- M1's training curves per team size (TensorBoard)

    python -m rl_search.scripts.multi_design.ma_mdp_check run --model rl_search/data/rl_real/m1_multi/ckpt/model_1200015.zip --label m1_1.2M --k 4 --case 0
    python -m rl_search.scripts.multi_design.ma_mdp_check figure --label m1_1.2M --k 4 --case 0
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

WORK = OUT/"ma_mdp_check"
TB = "rl_search/data/rl_real/m1_multi/tb"
GAMMA, LAM = 0.995, 0.95


def run(model_path: str, label: str, k: int, case_k: int, single: bool = False, sample: int | None = None, no_mates: bool = False):
    """``single``: a single-UAV model (R15) flown by every UAV on its own inputs, rewarded as in M1 (what M1 started from). ``sample``: actions
    sampled as in training (this torch seed) instead of the deterministic ones. ``no_mates``: the multi-UAV model with its teammate inputs
    blanked (tail channels 0, the vector entries as for a UAV alone; the shared belief stays): what the teammate inputs do."""
    warnings.filterwarnings("ignore")
    import torch
    from stable_baselines3 import PPO
    from rl_search.rl_multi import MultiRealEnv, P_HARD
    from rl_search.rl_real import RealRLConfig
    from rl_search.rl_real_eval import _case
    from rl_search.scripts.rl_eval.signal_check import settings_of
    torch.set_num_threads(1)
    if sample is not None:
        label = f"{label}_s{sample}"
    if no_mates:
        label = f"{label}_nomates"
    c = json.loads(Path(RUN_CFG).read_text())
    # the training reward exactly: settings_of leaves out the look reward (training only, not in the score) and the cut cost is a config field
    s = settings_of(c) | ({"look_w": float(c["look_w"])} if c.get("look_w") else {})
    model = PPO.load(model_path, device="cpu")
    if sample is not None:
        torch.manual_seed(sample)                          # after the load (PPO.load seeds torch with the run's own seed)
    case, scen, _, _ = _case("v2j", 3, case_k)
    env = MultiRealEnv(k, RealRLConfig(region="jirisan120", settings=s, mask_cost=float(c.get("mask_cost") or 0.0)), seed=case_k)
    obs = env.reset(options=case.options(scen), gaps=_gaps(case_k, k))
    cfg = env.uavs[0].cfg
    pos_w, look_w, mask_cost = float(env.uavs[0].env.pos_w), float(env.uavs[0].look_w), float(cfg.mask_cost)
    rec = {key: [] for key in ("t", "active", "ended", "team", "share", "seen", "look", "cut", "pen", "viol", "reward", "value", "xy")}
    while not env.done:
        if single:
            obs = [env.base_observation(i) for i in range(k)]
        elif no_mates:
            from rl_search.rl_multi import EXTRA_NAMES
            ne = len(EXTRA_NAMES)
            for o in obs:
                for name, _, _ in env.maps:
                    o[name][-1] = 0.0
                o["vec"][-ne:] = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0], np.float32)
        batch = {key: np.stack([o[key] for o in obs]) for key in obs[0]}
        with torch.no_grad():
            ot, _ = model.policy.obs_to_tensor(batch)
            val = model.policy.predict_values(ot).cpu().numpy().ravel()
        acts, _ = model.predict(batch, deterministic=sample is None)
        rec["t"].append(env.T)
        rec["active"].append(env.active.tolist())
        rec["ended"].append(env.ended.tolist())
        rec["value"].append(val.tolist())
        rec["xy"].append([u.env.plant.pose[:2].tolist() for u in env.uavs])
        obs, rew, _, info = env.step([np.atleast_1d(a) for a in np.ravel(acts)])
        for key in ("team",):
            rec[key].append(float(info[key]))
        for key, src in (("share", "share"), ("seen", "seen"), ("look", "look"), ("cut", "cut"), ("pen", "penalty"), ("viol", "violation")):
            rec[key].append(np.asarray(info[src], float).tolist())
        rec["reward"].append(np.asarray(rew, float).tolist())
    out = dict(label=label, model=model_path, k=k, case=case_k, launch_t=env.launch_t.tolist(), found=float(env.found),
               reward_scale=float(cfg.reward_scale), pos_w=pos_w, look_w=look_w, mask_cost=mask_cost, p_hard=P_HARD, **rec)
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK/f"{label}_k{k}_c{case_k}.json").write_text(json.dumps(out))
    print(f"{label} k={k} case {case_k + 1}: team POS {env.found:.3f}, {len(rec['t'])} steps", flush=True)


def run_train(model_path: str, label: str, k: int, n: int, seed: int = 7):
    """``n`` team flights of k UAVs on M1's own training draws (Chungju 120 km pool, the run's launch rule), actions sampled as in training:
    per flight and UAV the reward parts (team POS: own share / teammates, A', safety penalty, filter and edge cuts), split at the arrival
    (the first time the UAV's own share is above zero) into transit and search."""
    warnings.filterwarnings("ignore")
    import torch
    from stable_baselines3 import PPO
    from rl_search.rl_multi import MultiRealEnv
    from rl_search.rl_real import RealRLConfig, pool_dir_multi
    from rl_search.scripts.rl_eval.signal_check import settings_of
    torch.set_num_threads(1)
    c = json.loads(Path(RUN_CFG).read_text())
    s = settings_of(c) | ({"look_w": float(c["look_w"])} if c.get("look_w") else {}) | {"scen_v2": True, "types": c.get("types") or "road",
                                                                                          "midflight": float(c.get("midflight") or 0.0)}
    region = c["regions"][0]
    cfg = RealRLConfig(region=region, stage=3, settings=s, multi=True, pool=pool_dir_multi(s, region), reuse=int(c.get("reuse") or 2),
                       mask_cost=float(c.get("mask_cost") or 0.0))
    model = PPO.load(model_path, device="cpu")
    torch.manual_seed(seed)                                # after the load (PPO.load seeds torch with the run's own seed)
    env = MultiRealEnv(k, cfg, seed=seed)
    sc, pw, lw, mc = float(cfg.reward_scale), float(env.uavs[0].env.pos_w), float(env.uavs[0].look_w), float(cfg.mask_cost)
    rows = []
    for f in range(n):
        obs = env.reset()
        rec = dict(team=[], share=[], look=[], pen=[], cut=[], rew=[], act=[], t=[])
        while not env.done:
            batch = {key: np.stack([o[key] for o in obs]) for key in obs[0]}
            acts, _ = model.predict(batch, deterministic=False)
            rec["act"].append(env.active.copy())
            rec["t"].append(env.T)
            obs, rew, _, info = env.step([np.atleast_1d(a) for a in np.ravel(acts)])
            for key, src in (("team", "team"), ("share", "share"), ("look", "look"), ("pen", "penalty"), ("cut", "cut")):
                rec[key].append(np.asarray(info[src], float))
            rec["rew"].append(np.asarray(rew, float))
        act = np.array(rec["act"])
        team, share, look, pen, cut, rew = (np.array(rec[key]) for key in ("team", "share", "look", "pen", "cut", "rew"))
        t = (np.array(rec["t"]) - rec["t"][0])/60.0
        for i in range(k):
            a = act[:, i]
            hit = np.flatnonzero(share[:, i] > 1e-6)
            arr = hit[0] if len(hit) else len(a)                 # arrival: its first own find
            pre = np.arange(len(a)) < arr
            own, all_ = sc*pw*share[:, i]*a, sc*pw*team*a
            edge = rew[:, i]*a - (sc*(pw*team + lw*look[:, i]) - mc*cut[:, i] - pen[:, i])*a
            rows.append(dict(flight=f, uav=i, k=k, found=float(env.found), arrive_min=float(t[arr]) if arr < len(t) else None,
                             own=float(own.sum()), mates=float((all_ - own).sum()), look=float((sc*lw*look[:, i]*a).sum()),
                             pen=float(-(pen[:, i]*a).sum()), pen_transit=float(-(pen[:, i]*a*pre).sum()), cut=float(-(mc*cut[:, i]*a).sum()),
                             edge=float(edge.sum()), ret=float((rew[:, i]*a).sum()),
                             pen_curve=np.round(np.cumsum(pen[:, i]*a), 2)[::12].tolist(), own_curve=np.round(np.cumsum(own), 2)[::12].tolist()))
        print(f"{label} train flight {f + 1}/{n} k={k}: team POS {env.found:.3f}, return per UAV "
              + ", ".join(f"{r['ret']:.0f} (pen {r['pen']:.0f}, transit {r['pen_transit']:.0f})" for r in rows[-k:]), flush=True)
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK/f"train_{label}_k{k}.json").write_text(json.dumps(dict(label=label, k=k, rows=rows)))


def gae(r, v, act, end_terminal: bool):
    """GAE of one slot over the whole flight. ``end_terminal`` False: as the trainer does (the world's last step is the only terminal; the
    steps after the UAV's own flight end are masked afterwards, but their values and zero rewards are in the sums); True: the UAV's flight
    end is a terminal (no value after it)."""
    n = len(r)
    adv = np.zeros(n)
    last_act = np.flatnonzero(act)[-1]
    g = 0.0
    for t in range(n - 1, -1, -1):
        if end_terminal and t > last_act:
            adv[t] = 0.0
            g = 0.0
            continue
        nxt = 0.0 if (t == n - 1 or (end_terminal and t == last_act)) else v[t + 1]
        delta = r[t] + GAMMA*nxt - v[t]
        g = delta + GAMMA*LAM*g*(0.0 if (t == n - 1 or (end_terminal and t == last_act)) else 1.0)
        adv[t] = g
    return adv


def tb_curves() -> dict:
    import glob
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    fs = sorted(glob.glob(f"{TB}/*/events*"))
    out = {}
    for f in fs:
        ea = EventAccumulator(f, size_guidance={"scalars": 0})
        ea.Reload()
        for tag in ea.Tags()["scalars"]:
            if tag.startswith("team_k") or tag in ("rollout/ep_rew_mean",):
                ev = ea.Scalars(tag)
                out.setdefault(tag, []).extend((e.step, e.value) for e in ev)
    return {key: np.array(sorted(v)) for key, v in out.items()}


def figure(label: str, k: int, case_k: int):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "AppleGothic"
    plt.rcParams["axes.unicode_minus"] = False
    d = json.loads((WORK/f"{label}_k{k}_c{case_k}.json").read_text())
    act = np.array(d["active"], bool)
    team = np.array(d["team"])
    share, seen, look = (np.array(d[key]) for key in ("share", "seen", "look"))
    cut, pen, viol, rew, val = (np.array(d[key]) for key in ("cut", "pen", "viol", "reward", "value"))
    sc, pw, lw, mc = d["reward_scale"], d["pos_w"], d["look_w"], d["mask_cost"]
    t = (np.array(d["t"]) - d["t"][0])/60.0
    cols = ["#2a6fdb", "#e07b39", "#3a9a5b", "#8e44ad"]
    fig, axs = plt.subplots(2, 3, figsize=(21, 11.5))

    # (가) where each UAV's return comes from
    ax = axs[0, 0]
    own = np.array([sc*pw*(share[:, i]*act[:, i]).sum() for i in range(k)])
    mates = np.array([sc*pw*(team*act[:, i]).sum() for i in range(k)]) - own
    a_look = np.array([sc*lw*(look[:, i]*act[:, i]).sum() for i in range(k)])
    rest = np.array([(rew[:, i]*act[:, i]).sum() for i in range(k)]) - own - mates - a_look        # filter / edge cuts, safety penalty
    x = np.arange(k)
    ax.bar(x, own, 0.6, color="#2a6fdb", label="팀 POS 중 자기 몫 (내가 없었으면 못 찾았을 것)")
    ax.bar(x, mates, 0.6, bottom=own, color="#9ec1f0", label="팀 POS 중 나머지 (동료가 찾은 것)")
    ax.bar(x, a_look, 0.6, bottom=own + mates, color="#e07b39", label="A′ (화면 중앙에 들어온 입자 수, 자기 비행만)")
    ax.bar(x, rest, 0.6, color="#c0392b", label="잘림·안전 벌점 (음수)")
    for i in range(k):
        tot = own[i] + mates[i] + a_look[i] + rest[i]
        ax.text(i, own[i] + mates[i] + a_look[i] + 1, f"합 {tot:.0f}\n동료 몫 {mates[i]/max(own[i] + mates[i], 1e-9)*100:.0f}%", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{i + 1}번째 이륙" for i in range(k)])
    ax.set_ylabel("한 비행의 보상 합 (보상 단위)")
    ax.set_title(f"(가) 기체별 보상이 어디서 오나 ({label}, {k}대, 사례 {case_k + 1})", fontsize=10)
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    ax.set_ylim(min(rest.min()*1.3, 0) - 2, (own + mates + a_look).max()*1.35)

    # (나) A' against what the UAV found itself, over time
    ax = axs[0, 1]
    for i in range(k):
        ax.plot(t, np.cumsum(sc*lw*look[:, i]*act[:, i]), "-", color=cols[i], lw=1.6, label=f"{i + 1}번째: A′ 누적")
        ax.plot(t, np.cumsum(sc*pw*share[:, i]*act[:, i]), "--", color=cols[i], lw=1.2, label=f"{i + 1}번째: 자기 몫 POS 누적")
    ax.set_xlabel("첫 이륙 후 (분)")
    ax.set_ylabel("누적 보상")
    ax.set_title("(나) A′(실선)와 자기가 실제로 찾은 몫(점선)", fontsize=10)
    ax.legend(fontsize=7, frameon=False, ncol=2)

    # (다) per step: A' share among the UAVs vs own share among the UAVs
    ax = axs[0, 2]
    tot_look = look.sum(1)
    tot_share = share.sum(1)
    m = (tot_look > 0) & (act.sum(1) == k)
    for i in range(k):
        lf = np.where(m, look[:, i]/np.maximum(tot_look, 1e-12), np.nan)
        sf = np.where(m & (tot_share > 0), share[:, i]/np.maximum(tot_share, 1e-12), np.nan)
        ax.bar(i - 0.2, np.nanmean(lf), 0.4, color=cols[i], alpha=0.55)
        ax.bar(i + 0.2, np.nanmean(sf), 0.4, color=cols[i])
    ax.axhline(1/k, color="#888", lw=0.8, ls=":")
    ax.set_xticks(np.arange(k))
    ax.set_xticklabels([f"{i + 1}번째\n(연한 = A′, 진한 = 자기 몫)" for i in range(k)], fontsize=8)
    ax.set_ylabel("팀 안에서 차지하는 비율 (모두 비행 중인 걸음)")
    ax.set_title("(다) A′는 고르게, 실제 찾은 몫은 한쪽으로? (점선 = 1/대수)", fontsize=10)

    # (라) value around each UAV's own flight end
    ax = axs[1, 0]
    for i in range(k):
        a = act[:, i]
        last = np.flatnonzero(a)[-1]
        sl = slice(max(0, last - 40), len(a))
        tt = (np.arange(len(a))[sl] - last)*5.0
        ax.plot(tt, val[sl, i], "-o", ms=2.5, color=cols[i], lw=1.2, label=f"{i + 1}번째 (끝난 뒤 {len(a) - 1 - last}걸음 대기)")
    ax.axvline(0, color="#888", lw=0.8, ls=":")
    ax.axhline(0, color="#888", lw=0.6)
    ax.set_xlabel("자기 배터리 끝 기준 (초, 0 = 마지막 비행 걸음)")
    ax.set_ylabel("가치 V(o)")
    ax.set_title("(라) 배터리 끝 전후의 가치: 끝난 뒤 값은 0이어야 맞음", fontsize=10)
    ax.legend(fontsize=8, frameon=False)

    # (마) advantages as the trainer computes them vs with the flight end as a terminal
    ax = axs[1, 1]
    for i in range(k):
        a = act[:, i]
        r = rew[:, i]*a
        A0, A1 = gae(r, val[:, i], a, False), gae(r, val[:, i], a, True)
        last = np.flatnonzero(a)[-1]
        sl = slice(max(0, last - 120), last + 1)
        tt = (np.arange(len(a))[sl] - last)*5.0
        ax.plot(tt, (A0 - A1)[sl], "-", color=cols[i], lw=1.4, label=f"{i + 1}번째")
    ax.axhline(0, color="#888", lw=0.6)
    ax.set_xlabel("자기 배터리 끝 기준 (초)")
    ax.set_ylabel("이득 차이 (지금 학습 − 끝을 종료로 처리)")
    ax.set_title("(마) 끝을 종료로 안 본 탓에 이득이 달라지는 크기", fontsize=10)
    ax.legend(fontsize=8, frameon=False)

    # (바) training curves per team size
    ax = axs[1, 2]
    try:
        tb = tb_curves()
        for kk, c in zip((2, 3, 4), cols[1:]):
            arr = tb.get(f"team_k{kk}/POS")
            if arr is not None and len(arr):
                s, v = arr[:, 0], arr[:, 1]
                w = max(1, len(v)//25)
                vs = np.convolve(v, np.ones(w)/w, mode="valid")
                ax.plot(s/1e6, v, ".", color=c, ms=2, alpha=0.3)
                ax.plot(s[w - 1:]/1e6, vs, "-", color=c, lw=1.8, label=f"{kk}대 팀 POS (학습 중, 충주)")
        ax.set_xlabel("M1 스텝 (백만)")
        ax.set_ylabel("팀 POS (학습 시나리오, 계획용 입자)")
        ax.legend(fontsize=8, frameon=False)
    except Exception as e:                                        # noqa: BLE001
        ax.text(0.5, 0.5, f"TensorBoard: {e}", ha="center", transform=ax.transAxes)
    ax.set_title("(바) M1 학습 중 팀 POS (점 = 롤아웃별, 선 = 이동 평균)", fontsize=10)
    fig.suptitle(f"다수 기체 MDP 점검: 지리산 사례 {case_k + 1}, {k}대, {label} (결정은 무작위성 없이)", fontsize=13)
    fig.savefig(OUT/f"ma_mdp_check_{label}_k{k}.png", dpi=100, bbox_inches="tight")
    L = [f"# MDP 점검 {label} {k}대 사례 {case_k + 1}", "", "| 기체 | 자기 몫 | 동료 몫 | 직접 본 것 | A′ | 안전 벌점 | 필터 잘림 | 동료 몫 비율 | A′ 비율 | 끝난 뒤 대기 걸음 | 끝난 뒤 V 평균 |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for i in range(k):
        a = act[:, i]
        last = np.flatnonzero(a)[-1]
        after = val[last + 1:, i]
        L.append(f"| {i + 1} | {own[i]:.1f} | {mates[i]:.1f} | {sc*pw*(seen[:, i]*a).sum():.1f} | {a_look[i]:.1f} | {-(pen[:, i]*a).sum():.1f} | "
                 f"{-(mc*cut[:, i]*a).sum():.1f} | {mates[i]/max(own[i] + mates[i], 1e-9)*100:.0f}% | "
                 f"{a_look[i]/max(own[i] + mates[i] + a_look[i], 1e-9)*100:.0f}% | {len(after)} | {after.mean() if len(after) else float('nan'):.2f} |")
    (OUT/f"ma_mdp_check_{label}_k{k}.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("run", "figure", "train"))
    ap.add_argument("--n", type=int, default=3, help="train: team flights on training draws")
    ap.add_argument("--model")
    ap.add_argument("--label", required=True)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--case", type=int, default=0)
    ap.add_argument("--single", action="store_true", help="a single-UAV model (R15): every UAV flies on its own inputs")
    ap.add_argument("--sample", type=int, default=None, help="sample the actions as in training, with this seed")
    ap.add_argument("--no-mates", action="store_true", help="blank the teammate inputs of a multi-UAV model")
    a = ap.parse_args()
    if a.what == "run":
        run(a.model, a.label, a.k, a.case, a.single, a.sample, a.no_mates)
    elif a.what == "train":
        run_train(a.model, a.label, a.k, a.n)
    else:
        figure(a.label, a.k, a.case)


if __name__ == "__main__":
    main()

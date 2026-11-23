"""Can the M2 observation tell whether a teammate is closing in, and how many UAVs the team has? (user 2026-11-23: the safety penalty from
250 m counts only while a pair closes in; can the model see that?)  No training of the policy: probes (small classifiers) on observations.

Data: teams of 2, 3, 4 UAVs on the three Jirisan validation cases (rl_multi.MultiRealEnv, design "m2", launch gaps 5-10 s), every UAV
flown by R15 8.0M on its single-UAV inputs with sampled actions (varied geometry), every second decision. Per UAV and decision: the M2
teammate channels (small maps 1.6 / 3.2 / 6.4 km: ③ claim and danger zone; large maps: plan B), the vector, and the truth about its
nearest airborne teammate: distance, closing (relative position . relative velocity < 0, the penalty's rule), closest approach in 20 s
if both fly straight; and the team size.
Probes (torch, CPU), trained on cases 1-2 and tested on case 3 (never seen):
  P1 closing from the danger channels only; P2 closing from danger + claim + vector; P3 team size (2 / 3 / 4) from every teammate input.
Scored by distance to the nearest teammate (the penalty works within 250 m, the danger channel shows 15 / 30 / 60 s of reach).

    python -m rl_search.scripts.multi_design.closing_probe --reps 2 --workers 3
Writes report/closing_probe.png and .json.
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
import warnings
from multiprocessing import get_context
from pathlib import Path

import numpy as np

OUT = Path("rl_search/data/real/amboy/rl_eval/report")
R15 = Path("rl_search/data/rl_real/r15_chungju")
D_NEAR = 3200.0                   # teammate within this: keep the sample (the 6.4 km map's half side)
BINS = (0, 250, 500, 1000, 1600, 3200)
RATE_MIN = 3.0                    # m/s: the distance changes at least this fast (closing / receding readable)


def collect(job):
    case_k, k, rep = job
    warnings.filterwarnings("ignore")
    import torch
    from stable_baselines3 import PPO
    from rl_search.rl_multi import MultiRealEnv, SMALL_SIDE
    from rl_search.rl_real import RealRLConfig
    from rl_search.rl_real_eval import _case
    from rl_search.scripts.rl_eval.signal_check import settings_of
    torch.set_num_threads(1)
    torch.manual_seed(100*case_k + 10*k + rep)
    s = settings_of(json.loads((R15/"config.json").read_text()))
    case, scen, _, _ = _case("v2j", 3, case_k)
    env = MultiRealEnv(k, RealRLConfig(region="jirisan120", settings=s), seed=100*case_k + 10*k + rep, design="m2")
    rng = np.random.default_rng(100*case_k + 10*k + rep)
    env.reset(options=case.options(scen), gaps=[float(x) for x in rng.uniform(5.0, 10.0, k - 1)])
    model = PPO.load(str(R15/"ckpt/model_8000000.zip"), device="cpu")
    small = [n for n, side, _ in env.maps if side <= SMALL_SIDE]
    large = [n for n, side, _ in env.maps if side > SMALL_SIDE]
    rows = dict(danger=[], claim=[], planb=[], vec=[], d=[], closing=[], rate=[], cpa_d=[], cpa_t=[], k=[], case=[])
    n = 0
    while not env.done:
        obs = [env.base_observation(i) for i in range(k)]
        batch = {key: np.stack([o[key] for o in obs]) for key in obs[0]}
        acts, _ = model.predict(batch, deterministic=False)
        if n % 2 == 0:
            act = np.flatnonzero(env.active)
            for i in act:
                mates = [j for j in act if j != i]
                if not mates:
                    continue
                pi = env.uavs[i].env.plant
                P = np.array([env.uavs[j].env.plant.pose[:2] for j in mates])
                dd = np.hypot(*(P - pi.pose[:2]).T)
                j = mates[int(np.argmin(dd))]
                d = float(dd.min())
                if d > D_NEAR:
                    continue
                pj = env.uavs[j].env.plant
                rel = np.asarray(pj.pose[:2]) - np.asarray(pi.pose[:2])
                vrel = pj.v*np.array([np.cos(pj.pose[2]), np.sin(pj.pose[2])]) - pi.v*np.array([np.cos(pi.pose[2]), np.sin(pi.pose[2])])
                from rl_search.rl_multi import cpa
                t_c, d_c = cpa(pi.pose[:2], pi.pose[2], pj.pose[:2], pj.pose[2], pi.v)
                ch = env.mate_channels(i)
                rows["danger"].append(np.round(np.stack([ch[nm][1] for nm in small])*255).astype(np.uint8))
                rows["claim"].append(np.round((np.stack([ch[nm][0] for nm in small]) + 1)/2*255).astype(np.uint8))
                pb = np.stack([ch[nm][0] for nm in large])
                rows["planb"].append(np.round((pb.reshape(len(large), 16, 4, 16, 4).mean((2, 4)) + 1)/2*255).astype(np.uint8))
                rows["vec"].append(obs[i]["vec"].astype(np.float32))
                rows["d"].append(d); rows["closing"].append(float(rel @ vrel) < 0.0); rows["rate"].append(float(rel @ vrel)/max(d, 1e-6))
                rows["cpa_d"].append(d_c); rows["cpa_t"].append(t_c if np.isfinite(t_c) else 999.0); rows["k"].append(k); rows["case"].append(case_k)
        env.step([np.atleast_1d(a) for a in np.ravel(acts)])
        n += 1
    return {kk: np.array(v) for kk, v in rows.items()}


def probe(X_tr, y_tr, X_te, n_out, epochs=20, seed=0):
    """A small CNN (+ a vector branch) trained with cross-entropy; returns the test-set class probabilities."""
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    torch.set_num_threads(6)
    maps_tr, vec_tr = X_tr
    if vec_tr is not None:                             # vector entries on one scale (train mean / std)
        mu, sd = vec_tr.mean(0), vec_tr.std(0) + 1e-6
        vec_tr, vec_te = (vec_tr - mu)/sd, (X_te[1] - mu)/sd
        X_te = (X_te[0], vec_te)
    maps_te, vec_te = X_te
    c = maps_tr.shape[1]
    nv = vec_tr.shape[1] if vec_tr is not None else 0

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.cnn = nn.Sequential(nn.Conv2d(c, 16, 5, 2, 2), nn.ReLU(), nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(),
                                     nn.Conv2d(32, 32, 3, 2, 1), nn.ReLU(), nn.AdaptiveAvgPool2d(4), nn.Flatten())
            self.head = nn.Sequential(nn.Linear(32*16 + nv, 64), nn.ReLU(), nn.Linear(64, n_out))

        def forward(self, m, v):
            f = self.cnn(m)
            return self.head(torch.cat([f, v], 1) if v is not None else f)
    net = Net()
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    M = torch.as_tensor(maps_tr, dtype=torch.float32)/255.0
    V = torch.as_tensor(vec_tr, dtype=torch.float32) if vec_tr is not None else None
    Y = torch.as_tensor(y_tr, dtype=torch.long)
    w = torch.as_tensor(len(Y)/(n_out*np.maximum(np.bincount(y_tr, minlength=n_out), 1)), dtype=torch.float32)   # balanced classes
    lossf = nn.CrossEntropyLoss(weight=w)
    for ep in range(epochs):
        perm = torch.randperm(len(Y))
        for b in range(0, len(Y), 128):
            idx = perm[b:b + 128]
            opt.zero_grad()
            loss = lossf(net(M[idx], V[idx] if V is not None else None), Y[idx])
            loss.backward(); opt.step()
    with torch.no_grad():
        Mt = torch.as_tensor(maps_te, dtype=torch.float32)/255.0
        Vt = torch.as_tensor(vec_te, dtype=torch.float32) if vec_te is not None else None
        return torch.softmax(net(Mt, Vt), 1).numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--data", type=Path, default=None, help="reuse saved samples (npz) instead of flying again")
    a = ap.parse_args()
    if a.data is not None:
        D = dict(np.load(a.data))
    else:
        jobs = [(c, k, r) for r in range(a.reps) for c in (0, 1, 2) for k in (2, 3, 4)]
        with get_context("fork").Pool(a.workers, maxtasksperchild=1) as pool:
            parts = pool.map(collect, jobs, chunksize=1)
        D = {kk: np.concatenate([p[kk] for p in parts if len(p["d"])]) for kk in parts[0]}
        np.savez_compressed(OUT/"closing_probe_data.npz", **D)
    tr, te = D["case"] != 2, D["case"] == 2
    yc = D["closing"].astype(int)
    # (2026-11-23 second pass) pairs flying side by side after the launch keep their distance (range rate ~ 0): "closing" is then a
    # coin flip nobody can read. Closing is asked only where the distance changes by >= RATE_MIN m/s; the penalty question separately
    strong = np.abs(D["rate"]) >= RATE_MIN
    trs, tes = tr & strong, te & strong
    p1 = probe((D["danger"][trs], None), yc[trs], (D["danger"][tes], None), 2)[:, 1]
    both = np.concatenate([D["danger"], D["claim"]], 1)
    p2 = probe((both[trs], D["vec"][trs]), yc[trs], (both[tes], D["vec"][tes]), 2)[:, 1]
    ypen = (((D["d"] < 250) & (yc == 1)) | ((D["cpa_t"] <= 20) & (D["cpa_d"] < 250))).astype(int)
    p4 = probe((D["danger"][tr], None), ypen[tr], (D["danger"][te], None), 2)[:, 1]
    yk = D["k"] - 2
    big = np.concatenate([D["danger"], D["claim"]], 1)
    planb_up = np.repeat(np.repeat(D["planb"], 4, 2), 4, 3)
    p3 = probe((np.concatenate([big, planb_up], 1)[tr], D["vec"][tr]), yk[tr], (np.concatenate([big, planb_up], 1)[te], D["vec"][te]), 3)
    d_te, c_te = D["d"][tes], yc[tes]
    out = dict(n_train=int(trs.sum()), n_test=int(tes.sum()), closing_share=float(yc[strong].mean()), by_bin=[],
               weak_share=float((~strong).mean()), weak_share_near=float((~strong)[D["d"] < 1000].mean()))
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        m = (d_te >= lo) & (d_te < hi)
        if m.sum() < 5:
            out["by_bin"].append(dict(bin=[lo, hi], n=int(m.sum()))); continue
        out["by_bin"].append(dict(bin=[lo, hi], n=int(m.sum()), base=float(max(c_te[m].mean(), 1 - c_te[m].mean())),
                                  p1=float(((p1[m] > .5) == c_te[m]).mean()), p2=float(((p2[m] > .5) == c_te[m]).mean())))
    out["penalty_steps_test"] = int(ypen[te].sum())
    yp = ypen[te]
    out["penalty"] = dict(n_pos=int(yp.sum()), n=int(len(yp)), recall=float(((p4 > .5) & (yp == 1)).sum()/max(yp.sum(), 1)),
                          precision=float(((p4 > .5) & (yp == 1)).sum()/max((p4 > .5).sum(), 1)), acc=float(((p4 > .5) == yp).mean()),
                          base=float(max(yp.mean(), 1 - yp.mean())))
    kpred = p3.argmax(1)
    out["team_size"] = dict(acc=float((kpred == yk[te]).mean()), confusion=np.array([[int(((yk[te] == t) & (kpred == p)).sum()) for p in range(3)]
                                                                                       for t in range(3)]).tolist())
    (OUT/"closing_probe.json").write_text(json.dumps(out, indent=1))
    figure(D, tes, p1, p2, out)
    print(json.dumps(out, indent=1))


def figure(D, te, p1, p2, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["AppleGothic", "sans-serif"]; plt.rcParams["axes.unicode_minus"] = False
    INK, MUTED, GRID, BG = "#1f1f1e", "#6b6b67", "#e4e3de", "#fcfcfb"
    fig = plt.figure(figsize=(19, 10), facecolor=BG)
    gs = fig.add_gridspec(2, 4)
    a = fig.add_subplot(gs[0, :2])
    bins = [b for b in out["by_bin"] if "p1" in b]
    x = np.arange(len(bins)); w = .27
    for j, (key, lab, col) in enumerate((("base", "찍기 (많은 쪽만 답함)", "#b9b8b2"), ("p1", "위험 구역 채널만", "#E69F00"),
                                         ("p2", "위험 구역 + 동료 영역 + 벡터", "#2a78d6"))):
        v = [b[key]*100 for b in bins]
        a.bar(x + (j - 1)*w, v, w - .02, color=col, label=lab)
        for xi, vi in zip(x + (j - 1)*w, v):
            a.text(xi, vi + 1, f"{vi:.0f}", ha="center", fontsize=8.5)
    a.set_xticks(x, [f"{b['bin'][0]}~{b['bin'][1]} m\n(n={b['n']})" for b in bins]); a.set_ylim(0, 105); a.set_ylabel("맞힌 비율 (%)", color=MUTED)
    a.set_title(f"가장 가까운 동료가 다가오는지 관측만으로 맞히기 (거리가 초당 {RATE_MIN:g} m 넘게 변하는 쌍, 처음 보는 사례 3번)", loc="left", fontsize=11.5, color=INK)
    a.legend(frameon=False, fontsize=9, loc="lower left")
    a = fig.add_subplot(gs[0, 2])
    cm = np.array(out["team_size"]["confusion"], float)
    a.imshow(cm/np.maximum(cm.sum(1, keepdims=True), 1), cmap="Blues", vmin=0, vmax=1)
    for i_ in range(3):
        for j_ in range(3):
            a.text(j_, i_, f"{int(cm[i_, j_])}", ha="center", va="center", fontsize=10, color=INK)
    a.set_xticks(range(3), ["2대", "3대", "4대"]); a.set_yticks(range(3), ["2대", "3대", "4대"])
    a.set_xlabel("맞힌 대수", color=MUTED); a.set_ylabel("실제 대수", color=MUTED)
    a.set_title(f"팀 대수 맞히기: {out['team_size']['acc']*100:.0f}% (찍기 약 33%)", loc="left", fontsize=11.5, color=INK)
    # examples: danger channel of the 1.6 km map, closing and receding, near teammates
    idx_te = np.flatnonzero(te)
    for col, (lab, want) in enumerate((("다가오는 중", 1), ("멀어지는 중", 0))):
        cand = idx_te[(D["closing"][idx_te] == want) & (D["d"][idx_te] < 600)]
        for r_, ii in enumerate(cand[:2] if len(cand) else []):
            ax = fig.add_subplot(gs[1, col*2 + r_])
            ax.imshow(D["danger"][ii][0], cmap="magma", vmin=0, vmax=255)
            ax.plot(31.5, 31.5, marker=(3, 0, 0), ms=12, color="#2a78d6")
            pos = np.flatnonzero(idx_te == ii)[0]
            ax.set_title(f"{lab}, 거리 {D['d'][ii]:.0f} m\n'다가옴' 확률 (위험 구역만) {p1[pos]:.2f}", fontsize=10.5, color=INK)
            ax.set_xticks([]); ax.set_yticks([])
    a = fig.add_subplot(gs[0, 3]); a.axis("off")
    pn = out["penalty"]
    a.text(0, .95, f"학습 표본 {out['n_train']:,}, 시험 표본 {out['n_test']:,} (거리 변화가 뚜렷한 쌍)\n"
                   f"거리가 거의 안 변하는 쌍(나란히 비행): 전체의 {out['weak_share']*100:.0f}%, 1 km 안에서는 {out['weak_share_near']*100:.0f}%\n\n"
                   f"벌점 조건(20초 안 250 m 안 접근) 맞히기, 위험 구역만:\n  정확도 {pn['acc']*100:.0f}% (찍기 {pn['base']*100:.0f}%),\n"
                   f"  실제 벌점 상황 {pn['n_pos']}개 중 {pn['recall']*100:.0f}% 알아챔, '벌점'이라 답한 것 중 {pn['precision']*100:.0f}% 맞음\n\n아래: 1.6 km 지도의 위험 구역 채널\n(가운데 파란 삼각형 = 나, 위쪽 = 내 진행 방향,\n밝은 곳 = 동료가 곧 갈 수 있는 곳)",
           va="top", fontsize=10.5, color=INK)
    for a_ in fig.axes:
        a_.set_facecolor(BG)
    fig.suptitle("탐침: M2 관측으로 '동료가 다가오는지'와 '팀 대수'를 알 수 있나 (R15로 2·3·4대 비행, 학습 없음)", x=.01, ha="left", fontsize=13.5, color=INK)
    fig.tight_layout()
    fig.savefig(OUT/"closing_probe.png", dpi=75, facecolor=BG)


if __name__ == "__main__":
    main()

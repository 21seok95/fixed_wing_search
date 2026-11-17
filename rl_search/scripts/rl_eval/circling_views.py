"""Why does the policy circle? Several views on the same flights (user 2026-11-17: find the cause; energy shown is not it,
circling_cause.py). Flights: the user's 15 Chungju web flights and the road-type validation cases of the trained maps (nval / bval,
k = 0, 3, 6, 9, 12), flown deterministically by R14 road 3.2M.

Per decision: the policy's mean action before the clip to [-1, 1] and its value estimate V(s), the state around the UAV
(circling_cause.local_state), and circling (net heading change of 330 deg or more within +-90 s).
  1. actions: is the circling a hard turn (|mean action| >= 1, a saturated command) or a moderate one?
  2. value: what does the critic expect while circling, against the remaining probability?
  3. inputs: every 30 s, the mean action again with one input group neutralised (map channels set to 0, vector entries to their
     level-flight / unchanged values): which inputs hold the turn?
  4. alternatives: when a circle is detected (looking back 3 min), the state is saved and the next 10 min flown three ways:
     the policy, straight on, and a rule toward the heaviest 1 km cell of the remaining probability (re-aimed every 5 s); POS gained each.

    python -m rl_search.scripts.rl_eval.circling_views fly --set chungju      (and --set val; one process each)
    python -m rl_search.scripts.rl_eval.circling_views figure
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np

OUT = Path("rl_search/data/real/amboy/rl_eval/report")
WORK = OUT/"circling_views"
MODEL = "rl_search/data/rl_real/r14_road/ckpt/model_3200000.zip"
VAL_CASES = [("nval", k) for k in (0, 3, 6, 9, 12)] + [("bval", k) for k in (0, 3, 6, 9, 12)]
CH = ("입자 위치", "예측 위치", "남은 POS (지도 안 비율)", "남은 POS (절대)", "가시도", "카메라")
GROUPS = ["ch0", "ch1", "ch2", "ch3", "ch4", "ch5", "m1600", "m6400", "m25600", "m51200", "m102400", "attitude", "energy", "prob_left", "scales"]
GN = {"ch0": "입자 위치 (모든 지도)", "ch1": "예측 위치 (모든 지도)", "ch2": "남은 POS 지도 안 비율 (모든 지도)", "ch3": "남은 POS 절대 (모든 지도)",
      "ch4": "가시도 (모든 지도)", "ch5": "카메라 (1.6 km)", "m1600": "1.6 km 지도 전부", "m6400": "6.4 km 지도 전부", "m25600": "25.6 km 지도 전부",
      "m51200": "51.2 km 지도 전부", "m102400": "102.4 km 지도 전부", "attitude": "기울기·회전 상태·진로 차 (벡터)", "energy": "남은 에너지 (벡터)",
      "prob_left": "남은 확률 (벡터)", "scales": "지도별 크기 (벡터)"}
CIRC_K = 18                                                   # +-90 s at 5 s decisions


def circling(yaw):
    u = np.unwrap(np.asarray(yaw, float))
    out = np.zeros(len(u), bool)
    for i in range(len(u)):
        a, b = max(0, i - CIRC_K), min(len(u) - 1, i + CIRC_K)
        out[i] = abs(u[b] - u[a]) >= np.radians(330)
    return out


def ablate(obs: dict, g: str) -> dict:
    o = {k: v.copy() for k, v in obs.items()}
    if g.startswith("ch"):
        c = int(g[2:])
        for k in o:
            if k != "vec" and o[k].shape[0] > c and not (c == 5 and k != "m1600"):
                o[k][c] = 0.0
    elif g.startswith("m"):
        if g in o:
            o[g][:] = 0.0
    elif g == "attitude":
        o["vec"][0:3] = 0.0
    elif g == "energy":
        o["vec"][3] = 1.0
    elif g == "prob_left":
        o["vec"][5] = 1.0
    elif g == "scales":
        o["vec"][8:] = 0.0
    return o


def policy_out(model, obs_list):
    import torch
    keys = obs_list[0].keys()
    batch = {k: np.stack([o[k] for o in obs_list]) for k in keys}
    with torch.no_grad():
        t, _ = model.policy.obs_to_tensor(batch)
        dist = model.policy.get_distribution(t)
        mean = dist.distribution.mean.cpu().numpy().ravel()
        val = model.policy.predict_values(t).cpu().numpy().ravel()
    return mean, val


def heaviest_target(env):
    e = env.env
    b = e.belief
    P = b.positions(e.t)
    p = b.w/b.n
    q = np.floor(P/1000.0).astype(int)
    keys, inv = np.unique(q, axis=0, return_inverse=True)
    mass = np.bincount(inv.ravel(), weights=p)
    j = int(np.argmax(mass))
    m = inv.ravel() == j
    return np.average(P[m], axis=0, weights=np.maximum(p[m], 1e-12))


def rollout(env, model, obs, how: str, n: int = 120) -> float:
    from rl_search.plants import wrap
    e = env.env
    pos0 = env.pos_cum
    term = trunc = False
    for _ in range(n):
        if how == "policy":
            a, _ = model.predict(obs, deterministic=True)
        elif how == "straight":
            a = np.zeros(1, np.float32)
        else:
            tx, ty = heaviest_target(env)
            x, y = e.plant.pose[:2]
            a = np.array([np.clip(wrap(np.arctan2(ty - y, tx - x) - e.chi_tgt)/e.turn, -1.0, 1.0)], np.float32)
        obs, r, term, trunc, info = env.step(a)
        if term or trunc:
            break
    return float(env.pos_cum - pos0)


ENV_KEYS = ("steps", "A_cum", "pos_cum", "bonus_cum", "mask_cum", "look_cum", "roll_rev", "_roll_side", "_save_at", "_window")


def fly_one(env, model, options, tag):
    from rl_search.scripts.rl_eval.circling_cause import local_state
    obs, info = env.reset(options=options)
    e = env.env
    rec, abl, onsets, alts = [], [], [], []
    yaws = []
    last_onset = -10**9
    term = trunc = False
    k = 0
    while not (term or trunc):
        mean, val = policy_out(model, [obs])
        if k % 6 == 0:
            m2, _ = policy_out(model, [ablate(obs, g) for g in GROUPS])
            abl.append([k, float(mean[0])] + [float(v) for v in m2])
        yaws.append(float(e.plant.pose[2]))
        # a circle seen looking back 3 min: try the alternatives from here (at most one every 5 min, 6 per flight)
        if len(yaws) > 2*CIRC_K and k - last_onset >= 60 and len(onsets) < 6:
            u = np.unwrap(yaws[-(2*CIRC_K + 1):])
            if abs(u[-1] - u[0]) >= np.radians(330):
                last_onset = k
                snap = e.snapshot()
                saved = {a: getattr(env, a) for a in ENV_KEYS}
                gains = {}
                for how in ("policy", "straight", "heaviest"):
                    e.restore(snap)
                    e._nested += 1
                    for a_, v_ in saved.items():
                        setattr(env, a_, v_)
                    gains[how] = rollout(env, model, env.observation(), how)
                e.restore(snap)
                for a_, v_ in saved.items():
                    setattr(env, a_, v_)
                obs = env.observation()
                ls = local_state(env)
                alts.append(dict(k=k, t_min=e.t/60, **gains, local1=ls[0], local3=ls[1], heavy=ls[2], heavy_km=ls[3]/1000, remaining=ls[4]))
                onsets.append(k)
        a = np.clip(mean, -1, 1)
        obs, r, term, trunc, info = env.step(a.astype(np.float32))
        ls = local_state(env)
        rec.append((e.t, *e.plant.pose, float(e.plant.phi), float(mean[0]), float(val[0]), float(info["POS"]), *ls))
        k += 1
    rec = np.array(rec)
    circ = circling(rec[:, 3])
    print(f"{tag}: POS {rec[-1, 7]:.3f}, circling {100*circ.mean():.0f}%, onsets {len(alts)}", flush=True)
    return dict(tag=tag, rec=rec.tolist(), abl=abl, alts=alts)


def fly(which: str):
    warnings.filterwarnings("ignore")
    from rl_search.rl_real_eval import _case, _env, _model
    from rl_search.scripts.rl_eval.signal_check import settings_of
    from rl_search.scripts.real_map.web_rl import run_dir
    WORK.mkdir(parents=True, exist_ok=True)
    model = _model(MODEL)
    settings = settings_of(json.loads((run_dir(Path(MODEL))/"config.json").read_text()))
    out = []
    if which == "chungju":
        from rl_search.scripts.rl_eval.circling_cause import build, web_runs
        for i, m in enumerate(web_runs()):
            case, scen = build(m)
            out.append(fly_one(_env(settings, "chungju"), model, case.options(scen), f"충주 {i + 1}"))
    else:
        for sp, k in VAL_CASES:
            case, scen, vres, hold = _case(sp, 3, k)
            out.append(fly_one(_env(settings, case.region), model, case.options(scen), f"{'암보이' if sp == 'nval' else '바스토'} {k + 1}"))
    (WORK/f"{which}.json").write_text(json.dumps(out))


def figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["AppleGothic", "NanumGothic", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False
    INK, MUTED, GRID, BG = "#1f1f1e", "#6b6b67", "#e4e3de", "#fcfcfb"
    SETS = {s: json.loads((WORK/f"{s}.json").read_text()) for s in ("chungju", "val") if (WORK/f"{s}.json").exists()}
    SN = {"chungju": "충주 (학습 안 한 지도) 15개", "val": "암보이·바스토 검증 도로형 10개"}
    SC = {"chungju": "#d6336c", "val": "#2a78d6"}
    fig = plt.figure(figsize=(26, 22), facecolor=BG)
    gs = fig.add_gridspec(4, 4, height_ratios=[1, 1, 1.25, 1])
    summary = {}
    # row 1: circling by time, actions
    ax = fig.add_subplot(gs[0, 0])
    for s, runs in SETS.items():
        t = np.concatenate([np.array(r["rec"])[:, 0]/60 for r in runs]); c = np.concatenate([circling(np.array(r["rec"])[:, 3]) for r in runs])
        bins = np.arange(0, 61, 5); kk = np.digitize(t, bins) - 1
        ax.plot(bins[:-1] + 2.5, [100*c[kk == j].mean() if (kk == j).any() else np.nan for j in range(12)], marker="o", lw=2, color=SC[s],
                label=f"{SN[s]} (전체 {100*c.mean():.0f}%)")
        summary[s] = dict(circling=float(c.mean()))
    ax.set_title("① 원 선회 비율 (5분마다)", loc="left", fontsize=11, color=INK); ax.set_xlabel("신고 뒤 분", fontsize=9, color=MUTED)
    ax.legend(fontsize=8.5, frameon=False)
    ax = fig.add_subplot(gs[0, 1])
    for s, runs in SETS.items():
        A = np.concatenate([np.array(r["rec"]) for r in runs]); c = np.concatenate([circling(np.array(r["rec"])[:, 3]) for r in runs])
        m = A[:, 5]
        bb = np.linspace(-3, 3, 61)
        ax.hist(m[c], bins=bb, density=True, histtype="step", lw=2, color=SC[s], label=f"{SN[s]}: 원 선회 중")
        ax.hist(m[~c], bins=bb, density=True, histtype="step", lw=1.5, ls="--", color=SC[s], label=f"{SN[s]}: 아닐 때")
        summary[s]["sat_circ"] = float((np.abs(m[c]) >= 1).mean()); summary[s]["sat_not"] = float((np.abs(m[~c]) >= 1).mean())
    ax.axvline(-1, color=INK, lw=0.8, ls=":"); ax.axvline(1, color=INK, lw=0.8, ls=":")
    ax.set_title("② 정책의 평균 행동 (자르기 전; ±1 밖 = 최대 선회로 잘림)", loc="left", fontsize=11, color=INK)
    ax.set_xlabel("평균 행동 (−1 왼쪽 최대 … +1 오른쪽 최대)", fontsize=9, color=MUTED); ax.legend(fontsize=7.5, frameon=False)
    ax = fig.add_subplot(gs[0, 2])
    for s, runs in SETS.items():
        A = np.concatenate([np.array(r["rec"]) for r in runs]); c = np.concatenate([circling(np.array(r["rec"])[:, 3]) for r in runs])
        ax.scatter(np.log10(np.maximum(A[~c, 9], 1e-5)), A[~c, 6], s=3, alpha=0.25, color="#9a9a95")
        ax.scatter(np.log10(np.maximum(A[c, 9], 1e-5)), A[c, 6], s=3, alpha=0.35, color=SC[s], label=f"{SN[s]}: 원 선회 중")
    ax.set_xlabel("기체 3 km 안 남은 확률 (log10)", fontsize=9, color=MUTED); ax.set_ylabel("비평가의 가치 V(s)", fontsize=9, color=MUTED)
    ax.set_title("③ 가치 예측과 주변 남은 확률 (회색 = 원 선회 아님)", loc="left", fontsize=11, color=INK); ax.legend(fontsize=8, frameon=False)
    ax = fig.add_subplot(gs[0, 3])
    for s, runs in SETS.items():
        A = np.concatenate([np.array(r["rec"]) for r in runs]); c = np.concatenate([circling(np.array(r["rec"])[:, 3]) for r in runs])
        ax.scatter(np.log10(np.maximum(A[~c, 10], 1e-5)), A[~c, 6], s=3, alpha=0.25, color="#9a9a95")
        ax.scatter(np.log10(np.maximum(A[c, 10], 1e-5)), A[c, 6], s=3, alpha=0.35, color=SC[s], label=SN[s])
    ax.set_xlabel("지도 전체에서 가장 무거운 1 km 칸의 남은 확률 (log10)", fontsize=9, color=MUTED)
    ax.set_title("④ 가치 예측과 '어딘가에 몰린 확률'", loc="left", fontsize=11, color=INK); ax.legend(fontsize=8, frameon=False)
    # row 2: examples (time series) - two per set
    ex = [(s, r) for s, runs in SETS.items() for r in runs[:2]]
    for col, (s, r) in enumerate(ex[:4]):
        ax = fig.add_subplot(gs[1, col])
        A = np.array(r["rec"]); c = circling(A[:, 3]); t = A[:, 0]/60
        ax.fill_between(t, -3, 3, where=c, color=SC[s], alpha=0.12, lw=0, label="원 선회 중")
        ax.plot(t, A[:, 5], color=INK, lw=0.8, label="평균 행동")
        ax2 = ax.twinx()
        ax2.plot(t, A[:, 7], color="#1baf7a", lw=1.6, label="찾은 POS"); ax2.plot(t, np.log10(np.maximum(A[:, 9], 1e-5))/5 + 1, color="#eb6834", lw=1.2,
                                                                         label="3 km 안 남은 확률 (log, 눈금 조정)")
        ax2.set_ylim(0, 1.05)
        ax.set_ylim(-3, 3); ax.set_title(f"⑤ {r['tag']}: 행동·POS·주변 확률", loc="left", fontsize=10.5, color=INK)
        ax.set_xlabel("신고 뒤 분", fontsize=9, color=MUTED)
        if col == 0:
            h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels(); ax.legend(h1 + h2, l1 + l2, fontsize=7.5, frameon=False, loc="lower left")
    # row 3: input ablation
    ax = fig.add_subplot(gs[2, 0:3])
    w = 0.2
    pos = np.arange(len(GROUPS))
    j = 0
    for s, runs in SETS.items():
        rows, cflag = [], []
        for r in runs:
            A = np.array(r["rec"]); c = circling(A[:, 3])
            for a in r["abl"]:
                kk = int(a[0])
                if kk < len(c):
                    rows.append(a[1:]); cflag.append(c[kk])
        R = np.array(rows); cf = np.array(cflag, bool)
        base = R[:, 0]
        for flag, ls_, nm in ((True, "원 선회 중", 0), (False, "아닐 때", 1)):
            M = R[cf == flag]
            # share of decisions where the hard turn stops: |mean| >= 0.8 before, < 0.5 after removing the group
            hard = np.abs(M[:, 0]) >= 0.8
            stop = [(np.abs(M[hard, 1 + g]) < 0.5).mean() if hard.any() else np.nan for g in range(len(GROUPS))]
            ax.bar(pos + (j - 1.5)*w, 100*np.array(stop), width=w, color=SC[s], alpha=1.0 if flag else 0.4,
                   label=f"{SN[s]}: {ls_} (강한 선회 {hard.sum()}결정)")
            j += 1
            summary[s][f"stop_{'circ' if flag else 'not'}"] = {GROUPS[g]: float(stop[g]) for g in range(len(GROUPS))}
    ax.set_xticks(pos, [GN[g] for g in GROUPS], rotation=35, ha="right", fontsize=8.5)
    ax.set_ylabel("강한 선회(|행동| ≥ 0.8)가 멈추는 비율 (%)\n(그 입력을 없앴을 때 |행동| < 0.5)", fontsize=9, color=MUTED)
    ax.set_title("⑥ 입력 하나를 없애면(지도 채널 0, 벡터는 수평 비행·가득 등) 강한 선회가 멈추나: 높을수록 그 입력이 선회를 붙잡고 있음", loc="left", fontsize=11,
                 color=INK)
    ax.legend(fontsize=8, frameon=False, ncol=2)
    ax = fig.add_subplot(gs[2, 3])
    for s, runs in SETS.items():
        rows, cflag = [], []
        for r in runs:
            A = np.array(r["rec"]); c = circling(A[:, 3])
            for a in r["abl"]:
                if int(a[0]) < len(c):
                    rows.append(a[1:]); cflag.append(c[int(a[0])])
        R = np.array(rows); cf = np.array(cflag, bool)
        d = [np.mean(np.abs(R[cf, 1 + g] - R[cf, 0])) for g in range(len(GROUPS))]
        ax.barh(np.arange(len(GROUPS)) + (0.2 if s == "val" else -0.2), d, height=0.38, color=SC[s], label=SN[s])
    ax.set_yticks(range(len(GROUPS)), [GN[g] for g in GROUPS], fontsize=8); ax.invert_yaxis()
    ax.set_xlabel("원 선회 중 평균 행동의 변화 |Δ|", fontsize=9, color=MUTED)
    ax.set_title("⑦ 입력별 행동 변화 크기 (원 선회 중)", loc="left", fontsize=11, color=INK); ax.legend(fontsize=8, frameon=False)
    # row 4: alternatives at circle onsets
    ax = fig.add_subplot(gs[3, 0:2])
    xs = 0
    ticks = []
    for s, runs in SETS.items():
        al = [a for r in runs for a in r["alts"]]
        if not al:
            continue
        for i_h, (how, col, nm) in enumerate((("policy", INK, "정책 그대로"), ("straight", "#9a9a95", "직진"), ("heaviest", "#1baf7a", "가장 무거운 칸으로"))):
            v = np.array([a[how] for a in al])
            ax.boxplot(v, positions=[xs + i_h], widths=0.6, showfliers=False, patch_artist=True, boxprops=dict(facecolor=col, alpha=0.35))
            ax.scatter(np.full(len(v), xs + i_h) + np.random.default_rng(i_h).uniform(-0.2, 0.2, len(v)), v, s=10, color=col)
            ticks.append((xs + i_h, f"{nm}\n{SN[s][:6]}"))
        summary[s]["alts"] = dict(n=len(al), **{h: float(np.mean([a[h] for a in al])) for h in ("policy", "straight", "heaviest")},
                                  heaviest_better=float(np.mean([a["heaviest"] > a["policy"] + 0.005 for a in al])))
        xs += 4
    ax.set_xticks([t for t, _ in ticks], [l for _, l in ticks], fontsize=8.5)
    ax.set_ylabel("다음 10분 동안 더 찾은 POS", fontsize=9, color=MUTED)
    ax.set_title("⑧ 원 선회가 보인 순간부터 10분: 정책 / 직진 / 가장 무거운 칸으로 가는 규칙", loc="left", fontsize=11, color=INK)
    ax = fig.add_subplot(gs[3, 2:4])
    for s, runs in SETS.items():
        al = [a for r in runs for a in r["alts"]]
        if al:
            ax.scatter([a["heavy_km"] for a in al], [a["heaviest"] - a["policy"] for a in al], s=25, color=SC[s], label=SN[s])
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xlabel("원 선회 순간 가장 무거운 칸까지 거리 (km)", fontsize=9, color=MUTED); ax.set_ylabel("규칙 − 정책 (10분 POS)", fontsize=9, color=MUTED)
    ax.set_title("⑨ 규칙이 더 찾은 만큼 (0 위 = 정책이 더 나은 곳을 두고 돌았음)", loc="left", fontsize=11, color=INK); ax.legend(fontsize=8, frameon=False)
    for a_ in fig.axes:
        a_.set_facecolor(BG); a_.tick_params(colors=MUTED, labelsize=8.5); a_.grid(color=GRID, lw=.5)
        for s_ in a_.spines.values():
            s_.set_color(GRID)
    fig.suptitle("원 선회 원인 찾기 (R14 도로형): 행동, 가치 예측, 입력 하나씩 없애기, 원 선회 순간의 다른 선택", x=0.01, ha="left", fontsize=14, color=INK)
    fig.tight_layout()
    fig.savefig(OUT/"circling_views.png", dpi=72, facecolor=BG)
    (WORK/"summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=("fly", "figure"))
    ap.add_argument("--set", default="chungju", choices=("chungju", "val"))
    a = ap.parse_args()
    if a.what == "fly":
        fly(a.set)
    else:
        figure()


if __name__ == "__main__":
    main()

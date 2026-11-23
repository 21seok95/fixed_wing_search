"""Where does a team of UAVs gain over one, and how much of it does each reward form let the learner see? (user 2026-11-23: five UAVs
reach POS 0.6 even flying one UAV's path together; what a team adds is finding the spread, low-weight remainder late in the flight, so a
late find should count for more). No training: the saved multi-UAV flights on the Jirisan validation cases (R15 6.8M flown by 1-4 UAVs
on their own, and M1 1.2M, the team-reward run).

1. Team POS over the flight, and the gain over one UAV per 15 min block.
2. The remainder: at 15 / 30 / 45 min, how much probability is left and how spread it is (effective branches of the remaining weight,
   500 m pieces; the 90 % radius around its centre).
3. Reward forms on the same flights (team POS every 5 s): the share of the return seen from a decision at the launch and at 10 min that
   comes from the finds after 30 min (seen from the first find and from 20 min; nothing is found in the first ~10 min of transit), for
     P   reward = POS found this step, gamma 0.995 (now)
     A   the same, gamma 0.999
     B   reward = -log(remaining after / remaining before) (about found / remaining: the same find counts more when less is left), 0.995
     BA  B with gamma 0.999
   and the weight each form gives to a find at minute t relative to the same find at the launch.

    python -m rl_search.scripts.multi_design.reward_form_check
Writes report/reward_form_check.png and .json.
"""
from __future__ import annotations

import base64
import gzip
import json
from pathlib import Path

import numpy as np

OUT = Path("rl_search/data/real/amboy/rl_eval/report")
RUNS = Path("rl_search/data/real/chungju/web_runs")
SETS = {"R15 각자": "r15_6p8M", "M1 120만 (팀 보상)": "m1_1p2M"}
FORMS = {"P 지금 (찾은 확률, γ 0.995)": ("pos", 0.995), "A 찾은 확률, γ 0.999": ("pos", 0.999),
         "B 남은 확률 대비, γ 0.995": ("rel", 0.995), "BA 남은 확률 대비, γ 0.999": ("rel", 0.999)}
MIN = (15, 30, 45, 60)


def load(rid):
    r = json.load(gzip.open(RUNS/rid/"run.json.gz", "rt"))
    o, res, meta = r["opt"], r["res"], r["meta"]
    t0 = float(meta["case_info"].get("t_launch_s", 0.0))
    t = np.asarray(o["pos_t"], float) - t0
    pos = np.asarray(o["pos"], float)
    return dict(t=t, pos=pos, esc=float(o.get("escaped", 0.0)), r=r, t0=t0, hold=meta["holdout"]["pos"])


def remainder(d, minutes):
    """(remaining probability, effective branches, 90 % radius km) of the team's remaining weight at these minutes after the launch."""
    from rl_search.scripts.rl_eval.branch_check import branches
    o, res = d["r"]["opt"], d["r"]["res"]
    N = res["N"]
    W = np.frombuffer(base64.b64decode(o["weight"]), np.uint8).reshape(o["T"], N).astype(float)/255.0
    P = np.frombuffer(base64.b64decode(res["pos"]), "<i2").reshape(res["T"], N, 2).astype(float)*res["pos_unit_m"]
    st = np.frombuffer(base64.b64decode(res["state"]), np.uint8).reshape(res["T"], N)
    rng = np.random.default_rng(0)
    out = []
    for m in minutes:
        f = int(round((d["t0"] + 60*m)/o["dt_s"]))
        if f >= min(o["T"], res["T"]):
            out.append((np.nan, np.nan, np.nan)); continue
        w = W[f]*(st[f] != 4)
        if w.sum() <= 0:
            out.append((0.0, 0.0, 0.0)); continue
        pick = rng.choice(N, 4000, p=w/w.sum())
        _, eff = branches(P[f, pick])
        c = np.average(P[f], axis=0, weights=w)
        dist = np.hypot(*(P[f] - c).T)
        o_ = np.argsort(dist); cw = np.cumsum(w[o_])/w.sum()
        out.append((float(w.sum()/N), eff, float(dist[o_][np.searchsorted(cw, .9)]/1000)))
    return out


def form_weights(t, pos, kind, gamma, t_from):
    """Per-step reward of the form and its discount weight seen from t_from (s after the launch)."""
    gain = np.diff(pos)
    rem = 1.0 - pos[:-1]
    if kind == "pos":
        r = gain
    else:
        r = -np.log(np.clip((rem - gain)/np.maximum(rem, 1e-9), 1e-9, 1.0))
    tt = t[1:]
    disc = np.where(tt >= t_from, gamma**((tt - t_from)/5.0), 0.0)
    return tt, r, disc


def main():
    rows = {}
    for name, tag in SETS.items():
        for k in (1, 2, 3, 4):
            ds = [load(f"eval-v2j-s3-k{c}-ma{k}-{tag}") for c in range(3) if (RUNS/f"eval-v2j-s3-k{c}-ma{k}-{tag}/run.json.gz").exists()]
            if ds:
                rows[(name, k)] = ds
    summary = dict(pos={}, remainder={}, forms={})
    for (name, k), ds in rows.items():
        at = [[float(np.interp(60*m, d["t"], d["pos"])) for m in MIN] for d in ds]
        summary["pos"][f"{name} {k}대"] = dict(at_min=np.mean(at, 0).round(4).tolist(), holdout=float(np.mean([d["hold"] for d in ds])))
        summary["remainder"][f"{name} {k}대"] = np.nanmean([remainder(d, (15, 30, 45)) for d in ds], 0).round(3).tolist()
        fr = {}
        for fname, (kind, g) in FORMS.items():
            late = []
            for which in ("first", 1200.0):            # from the first find (POS > 0.005) and from 20 min (no find before 10 min: transit)
                sh = []
                for d in ds:
                    t_from = float(d["t"][np.argmax(d["pos"] > 0.005)]) if which == "first" else which
                    tt, r, disc = form_weights(d["t"], d["pos"], kind, g, t_from)
                    tot = (r*disc).sum()
                    sh.append(float((r*disc)[tt >= 1800].sum()/tot) if tot > 0 else np.nan)
                late.append(float(np.nanmean(sh)))
            fr[fname] = late
        summary["forms"][f"{name} {k}대"] = fr
    figure(rows, summary)
    (OUT/"reward_form_check.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print(json.dumps(summary, ensure_ascii=False, indent=1))


def figure(rows, S):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["AppleGothic", "sans-serif"]; plt.rcParams["axes.unicode_minus"] = False
    INK, MUTED, GRID, BG = "#1f1f1e", "#6b6b67", "#e4e3de", "#fcfcfb"
    KC = {1: "#6b6b67", 2: "#2a78d6", 3: "#E69F00", 4: "#d6336c"}
    fig, ax = plt.subplots(2, 3, figsize=(21, 11), facecolor=BG)
    # 1a POS curves (R15 on their own), mean of the cases
    a = ax[0, 0]
    tg = np.arange(0, 3601, 30.0)
    for (name, k), ds in rows.items():
        y = np.mean([np.interp(tg, d["t"], d["pos"]) for d in ds], 0)
        a.plot(tg/60, y, color=KC[k], lw=2 if name.startswith("R15") else 1.2, ls="-" if name.startswith("R15") else "--",
               label=f"{name} {k}대 (최종 {y[-1]:.2f})")
    a.set_xlabel("이륙 뒤 분", color=MUTED); a.set_ylabel("팀 POS (계획 입자, 사례 3개 평균)", color=MUTED)
    a.set_title("팀 POS: 실선 = R15를 각자, 점선 = M1 (팀 보상 학습)", loc="left", fontsize=11.5, color=INK)
    a.legend(frameon=False, fontsize=8.5, ncol=2)
    # 1b gain over one UAV per 15 min block (R15 on their own)
    a = ax[0, 1]
    base = np.array(S["pos"]["R15 각자 1대"]["at_min"])
    blocks = np.diff(np.r_[0, base])
    w = .2
    for j, k in enumerate((1, 2, 3, 4)):
        v = np.diff(np.r_[0, np.array(S["pos"][f"R15 각자 {k}대"]["at_min"])])
        a.bar(np.arange(4) + (j - 1.5)*w, v, w - .02, color=KC[k], label=f"{k}대")
        for i, x in enumerate(v):
            a.text(i + (j - 1.5)*w, x + .003, f"{x:.2f}", ha="center", fontsize=7.5)
    a.set_xticks(range(4), ["0~15분", "15~30분", "30~45분", "45~60분"])
    a.set_ylabel("그 15분 동안 찾은 확률", color=MUTED)
    a.set_title("15분마다 찾은 확률 (R15를 각자, 1~4대)", loc="left", fontsize=11.5, color=INK); a.legend(frameon=False, fontsize=9)
    # 1c remainder spread
    a = ax[0, 2]
    for j, k in enumerate((1, 2, 3, 4)):
        rem = np.array(S["remainder"][f"R15 각자 {k}대"])
        a.plot([15, 30, 45], rem[:, 1], "o-", color=KC[k], label=f"{k}대: 남은 확률 {' / '.join(f'{x:.2f}' for x in rem[:, 0])}")
    a.set_xlabel("이륙 뒤 분", color=MUTED); a.set_ylabel("남은 확률의 유효 갈래 수 (500 m 조각)", color=MUTED)
    a.set_title("남은 확률은 얼마나 흩어져 있나 (R15를 각자; 범례 = 15 / 30 / 45분에 남은 확률)", loc="left", fontsize=11, color=INK)
    a.legend(frameon=False, fontsize=8.5)
    # 2a weight of a find at minute t relative to the launch (typical 4-UAV flight)
    a = ax[1, 0]
    d = rows[("R15 각자", 4)][0]
    FC = dict(zip(FORMS, ["#6b6b67", "#2a78d6", "#E69F00", "#d6336c"]))
    for fname, (kind, g) in FORMS.items():
        tt = d["t"][1:]
        rem = 1.0 - d["pos"][:-1]
        wgt = g**(tt/5.0)*(1.0/np.maximum(rem, 1e-3) if kind == "rel" else 1.0)
        a.plot(tt/60, wgt/wgt[0], color=FC[fname], lw=2, label=fname)
    a.set_yscale("log"); a.set_xlabel("이륙 뒤 분", color=MUTED)
    a.set_ylabel("같은 확률을 찾았을 때 보상의 무게 (이륙 때 = 1)", color=MUTED)
    a.set_title("후반의 같은 0.01이 얼마나 무겁게 보이나 (이륙 때 판단 기준, 지리산 1번 4대)", loc="left", fontsize=11, color=INK)
    a.legend(frameon=False, fontsize=9)
    # 2b share of the return after 30 min, seen from the launch and from 10 min
    for col, (j, lab) in zip((1, 2), enumerate(("처음 찾기 시작한 때", "20분 때"))):
        a = ax[1, col]
        keys = [f"R15 각자 {k}대" for k in (1, 2, 3, 4)]
        w = .2
        for i, fname in enumerate(FORMS):
            v = [S["forms"][kk][fname][j] for kk in keys]
            a.bar(np.arange(4) + (i - 1.5)*w, np.array(v)*100, w - .02, color=FC[fname], label=fname)
            for x_, y_ in zip(np.arange(4) + (i - 1.5)*w, v):
                a.text(x_, y_*100 + .8, f"{y_*100:.0f}", ha="center", fontsize=7.5)
        a.set_xticks(range(4), ["1대", "2대", "3대", "4대"])
        a.set_ylabel("%", color=MUTED)
        a.set_title(f"{lab}의 판단에서 본 보상 중 30분 이후 몫 (R15를 각자)", loc="left", fontsize=11.5, color=INK)
        a.legend(frameon=False, fontsize=8.5)
    for a_ in ax.ravel():
        a_.set_facecolor(BG); a_.grid(color=GRID, lw=.5); a_.tick_params(colors=MUTED)
        for sp in a_.spines.values():
            sp.set_color(GRID)
    fig.suptitle("다수 기체의 이득은 언제 나오나, 보상 형태별로 학습은 후반을 얼마나 보나 (지리산 검증 3개, 학습 없음)", x=.01, ha="left", fontsize=13.5, color=INK)
    fig.tight_layout()
    fig.savefig(OUT/"reward_form_check.png", dpi=72, facecolor=BG)


if __name__ == "__main__":
    main()

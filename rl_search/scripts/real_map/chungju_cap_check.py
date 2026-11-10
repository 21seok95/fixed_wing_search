"""Which tasks hit the 12 min task cap, and why (step 3, lane time flown, 10 min transit limit).

The cap: t_cap = plan time + planned transit (straight line to the lane start) + search_s (600 s) + extra_s (120 s).
For every capped task: the final rectangle's lane time (dry flight), the same rectangle with one pass fewer
(the "spacing" move of the local search adds passes k without checking the 10 min budget), and the flown lane time.

Needs step3/timing_flown_<seed>.npz (chungju_timing_check --lane-time flown).
Figure: rl_search/data/real/chungju/step3_flown/capped_tasks.png

    python -m rl_search.scripts.real_map.chungju_cap_check
"""
from __future__ import annotations

import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rl_search.real_terrain import DATA
from rl_search.scripts.real_map.chungju_step3 import LEAD_M, OUT

plt.rcParams["font.family"] = ["AppleGothic", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False


def main():
    warnings.filterwarnings("ignore")
    from rl_search.accordion import flown_lane_time
    from rl_search.real_scenario import BANK_DEG, COVER_M, make_env
    env = make_env(1, lead_m=LEAD_M, bank_deg=BANK_DEG)
    rows = []
    for s in range(1, 11):
        o = np.load(OUT/f"timing_flown_{s}.npz")
        for k, (task, p, rc) in enumerate(zip(o["tasks"], o["plan"], o["rects"])):
            t0, tl, te, cap = task
            if not np.isfinite(tl):
                continue
            n, sp = int(p[4]), p[5]
            kk = int(round(COVER_M/sp))
            units = max(1, int(round(n/kk)))
            fewer = flown_lane_time(env, rc[3], units*(kk - 1), COVER_M/(kk - 1)) if kk > 1 else np.nan
            rows.append(dict(seed=s, task=k, k=kk, n=n, ell=rc[3], wid=rc[4], plan=p[3], fewer=fewer,
                             real=(te - tl) if np.isfinite(te) else np.nan, cap=cap > 0, left=t0 + p[2] + 720 - tl))
    capped = [r for r in rows if r["cap"]]
    done = [r for r in rows if not r["cap"] and np.isfinite(r["real"])]

    fig, axs = plt.subplots(1, 2, figsize=(20, 7.5), layout="constrained", gridspec_kw=dict(width_ratios=[2.2, 1]))
    ax = axs[0]
    x = np.arange(len(capped))
    ax.bar(x - .27, [r["plan"]/60 for r in capped], .27, color="#c0392b", label="고른 사각형의 레인 시간 (모의 비행)")
    ax.bar(x, [r["fewer"]/60 for r in capped], .27, color="#2a4d7a", label="같은 사각형, 겹침 횟수 k − 1")
    ax.bar(x + .27, [r["real"]/60 for r in capped], .27, color="#7f8c8d", label="실제 난 레인 시간 (상한에서 끊김)")
    ax.axhline(10, color="k", ls=":", lw=1); ax.text(-.6, 10.15, "탐색 예산 10분", fontsize=9)
    ax.axhline(12, color="#c0392b", ls="--", lw=1); ax.text(-.6, 12.15, "상한 (이동 + 10분 + 여유 2분)", fontsize=9, color="#c0392b")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r['seed']}-{r['task'] + 1}\nk={r['k']}, {r['ell']/1000:.1f}×{r['wid']/1000:.2f} km" for r in capped], fontsize=8)
    ax.set_ylabel("분"); ax.legend(loc="upper right")
    ax.set_title(f"12분 상한에 걸린 작업 {len(capped)}/{len(rows)}개 (시나리오-작업): 모두 겹침 횟수 k를 1 줄이면 10분 안에 들어간다",
                 fontsize=11)
    ax = axs[1]
    ks = np.arange(1, 4)
    ax.bar(ks - .2, [sum(r["k"] == k for r in rows) for k in ks], .4, color="#7f8c8d", label="전체 작업")
    ax.bar(ks + .2, [sum(r["k"] == k for r in capped) for k in ks], .4, color="#c0392b", label="상한에 걸린 작업")
    ax.set_xticks(ks); ax.set_xticklabels([f"k = {k}\n(간격 {190/k:.0f} m)" for k in ks])
    ax.set_ylabel("작업 수"); ax.legend()
    ax.set_title("한 점을 몇 번 훑나 (k, 레인 간격 = 190 m / k)", fontsize=11)
    fig.suptitle("12분 상한: 다듬기의 '간격' 동작이 예산을 확인하지 않고 겹침 횟수를 늘린다 (레인 시간 모의 비행, 이동 제한 10분, 시나리오 10개)",
                 fontsize=13)
    out = DATA/"chungju"/"step3_flown"/"capped_tasks.png"
    fig.savefig(out, dpi=95)
    print(f"tasks {len(rows)}, capped {len(capped)}; capped: planned {np.mean([r['plan'] for r in capped]):.0f} s, "
          f"k-1 {np.mean([r['fewer'] for r in capped]):.0f} s (fits 600 s: {sum(r['fewer'] <= 600 for r in capped)}), "
          f"flown share {np.mean([r['real']/r['plan'] for r in capped]):.2f}; not capped: planned {np.mean([r['plan'] for r in done]):.0f} s")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()

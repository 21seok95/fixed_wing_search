"""Web runs of the visibility-input test (circling_views, user 2026-11-17): the user's Chungju RL web flights flown again exactly as the web
viewer flies them (web_rl._run), except that the policy is shown visibility 1 (open ground) on every map cell inside the terrain; the
detection itself still uses the real visibility. Saved as web runs next to the originals (the running viewer lists them on its own).

    python -m rl_search.scripts.rl_eval.web_vis1_runs
"""
from __future__ import annotations

import time
import warnings

import numpy as np


class _Q:
    def put(self, m):
        pass


def main():
    warnings.filterwarnings("ignore")
    from rl_search.rl_real import RealRLEnv
    from rl_search.scripts.real_map import web_rl
    from rl_search.scripts.real_map.vehicle_web import run_meta, save_run
    from rl_search.scripts.rl_eval.circling_cause import web_runs
    orig = RealRLEnv.observation

    def observation(self):
        o = orig(self)
        for k in o:
            if k != "vec":
                o[k][4] = np.where(o[k][4] > 0, 1.0, o[k][4])
        return o
    RealRLEnv.observation = observation
    stamp = time.strftime("%Y%m%d-%H%M%S")
    for i, m in enumerate(web_runs()):
        if "가시도 입력 1" in m.get("label", ""):
            continue
        p = dict(view=m["view"], config=m["config"], model=m["model"], launch_min=m.get("launch_min", 0.0), kind="rl")
        t0 = time.perf_counter()
        data = web_rl._run(p, _Q())
        data["opt"]["sec"] = round(time.perf_counter() - t0, 1)
        extra = data.pop("meta_extra", {})
        rid = f"{stamp}-{900 + i:03d}"
        label = f"{m['label']} · 시험: 가시도 입력 1 (원래 {m['id'][9:15]})"
        meta = run_meta(data["opt"], rid, label, time.strftime("%m-%d %H:%M"), m["view"], m["config"], group="web", **extra)
        save_run(rid, data, meta)
        print(f"{i + 1}: {rid} POS {data['opt']['final']:.3f} (web {m['final']:.3f})", flush=True)


if __name__ == "__main__":
    main()

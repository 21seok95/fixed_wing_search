"""Start-point rules for the accordion search (docs/sarops_approach_todo.md S.10): densest cell, smoothed density,
multi-start over density peaks. Sensor new + lrc, 8 scenarios.

    python -m rl_search.scripts.check_accordion_start
"""
import warnings; warnings.filterwarnings("ignore")
import sys, numpy as np
from multiprocessing import Pool
from pathlib import Path
VARS={"base (densest 200 m cell)":{}, "smoothed start 500 m":dict(start_smooth_m=500), "smoothed start 1000 m":dict(start_smooth_m=1000),
      "multi-start 3 peaks":dict(n_starts=3), "multi-start 5 peaks":dict(n_starts=5), "multi-start 5, smooth 500":dict(n_starts=5,start_smooth_m=500)}
def run(seed):
    from rl_search.accordion import AccordionPlanner
    from rl_search.config import get_preset
    from rl_search.rl_env import RLConfig, RLSearchEnv
    from rl_search.scenario import make_scenario
    from rl_search.terrain import load_terrain
    from rl_search.scripts.check_accordion import SENSORS
    terr=load_terrain(Path("rl_search/data/terrain_seed7_60km.npz"))
    out={}
    for name,kw in VARS.items():
        env=RLSearchEnv(terr,get_preset("rc_cessna",SENSORS["new"]),RLConfig(stage=3,detect_model="lrc"),seed=seed); env.fast=True
        env.reset(options=dict(scenario=make_scenario(env.sampler,seed),stage=3))
        pol=AccordionPlanner("tavg_pca",cell=200,**kw); g=np.zeros(2); n=np.zeros(2)
        while True:
            a=pol(env); on=int(bool(pol.follower is not None and pol.follower.started))
            p0=env.pos_cum; _,_,term,_,_=env.step([a]); g[on]+=env.pos_cum-p0; n[on]+=1
            if term: break
        C=np.array([[r.cx,r.cy] for _,r,_ in pol.rects]); jumps=(np.hypot(*np.diff(C,axis=0).T)>3000).sum() if len(C)>1 else 0
        out[name]=(g.sum(), n[0]/n.sum(), jumps)
    return out
if __name__=="__main__":
    with Pool(2) as p: R=p.map(run, range(300,308))
    print("| start rule | POS mean | min | time not on lanes | jumps > 3 km per flight |\n|---|---|---|---|---|")
    for name in VARS:
        a=np.array([r[name] for r in R]); print(f"| {name} | {a[:,0].mean():.3f} | {a[:,0].min():.3f} | {a[:,1].mean():.0%} | {a[:,2].mean():.1f} |")
    print("per seed POS:", {n:[round(r[n][0],3) for r in R] for n in VARS})

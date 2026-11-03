"""Ideal (Koopman/Stone) effort allocation over the prior for 1-6 UAVs per 10-min cycle: how many separate regions get effort (docs/sarops_approach_todo.md S.8).

    python -m rl_search.scripts.check_split
"""
import numpy as np, warnings; warnings.filterwarnings("ignore")
from scipy.ndimage import gaussian_filter, label
from rl_search.config import get_preset
from rl_search.terrain import load_terrain
from rl_search.rl_env import RLSearchEnv, RLConfig
from rl_search.scenario import make_scenario
P=get_preset("rc_cessna"); terr=load_terrain("rl_search/data/terrain_seed7_60km.npz")
env=RLSearchEnv(terr,P,RLConfig(stage=3),seed=0)
cell=0.1  # km
W=0.3*0.115   # effective sweep width km (POD 0.3 x footprint width)
L=15*600*0.8/1000  # km flown in a 10-min cycle
def alloc(p, z):
    # Koopman/Stone: effort density phi = max(0, ln(p/lam)), sum phi*cellarea = z (exposure units)
    lo,hi=1e-12,p.max()
    for _ in range(80):
        lam=np.sqrt(lo*hi); f=np.maximum(0,np.log(np.maximum(p,1e-300)/lam)).sum()*cell*cell
        lo,hi=(lam,hi) if f>z else (lo,lam)
    phi=np.maximum(0,np.log(np.maximum(p,1e-300)/lam))
    return phi, lam
rows=[]
for seed in range(300,308):
    sc=make_scenario(env.sampler,seed)
    for tm in (20,35,50):
        k=int(tm*60/sc.save_dt_s); act=(sc.exit_step<0)|(sc.exit_step>k); pts=np.asarray(sc.track[k],float)[act]/1000
        ex=np.arange(pts[:,0].min()-1,pts[:,0].max()+1.1,cell); ey=np.arange(pts[:,1].min()-1,pts[:,1].max()+1.1,cell)
        H,_,_=np.histogram2d(pts[:,0],pts[:,1],bins=[ex,ey]); H=gaussian_filter(H,1.5)
        p=H/H.sum()/(cell*cell)*act.mean()   # probability density per km^2 (in-map mass)
        for N in (1,2,4,6):
            z=N*W*L
            phi,lam=alloc(p,z)
            lab,n=label(phi>0, structure=np.ones((3,3)))
            share=np.bincount(lab.ravel(), weights=phi.ravel())[1:]; share=share/share.sum()
            big=(share>=0.05).sum()
            pos=(p*(1-np.exp(-phi))).sum()*cell*cell
            area=(phi>0).sum()*cell*cell
            rows.append((seed,tm,N,big,share.max(),area,pos))
r=np.array(rows)
print("N UAVs | t(min) | regions with >=5% of effort (mean, max) | largest region share | allocated area km2 | ideal POS this cycle")
for N in (1,2,4,6):
    for tm in (20,35,50):
        m=r[(r[:,2]==N)&(r[:,1]==tm)]
        print(f"{N} | {tm} | {m[:,3].mean():.1f}, {int(m[:,3].max())} | {m[:,4].mean():.0%} | {m[:,5].mean():.2f} | {m[:,6].mean():.3f}")

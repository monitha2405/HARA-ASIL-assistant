"""Synthetic HARA dataset generator (no confidential data used).
Latent rules + 6% label noise mimic expert inconsistency."""
import numpy as np, pandas as pd
from .asil import asil
FUNCS = {"emergency braking":(3,2),"electric power steering assist":(3,2),"airbag deployment":(3,3),
 "adaptive cruise control":(2,1),"lane keeping assist":(2,1),"door locking":(1,0),"headlamp control":(1,1),
 "windscreen wiper":(1,1),"power window":(1,0),"infotainment display":(0,0),"park lock gear shift":(3,2),
 "battery management":(2,2),"traction control":(2,2),"tyre pressure monitoring":(1,0)}
MALS = {"loss of":(0,0),"unintended activation of":(0,1),"delayed response of":(0,0),
 "excessive output of":(0,1),"stuck output of":(0,0),"reversed output of":(1,1)}
SCEN = {"highway cruising":(90,130,15,40,.0),"urban driving":(20,50,10,30,.5),"parking maneuver":(2,10,2,8,.4),
 "standing at a traffic light":(0,0,5,15,.3),"wet road cornering":(50,90,3,9,.05),
 "night driving on rural road":(60,90,5,15,.1),"towing a trailer downhill":(30,60,.2,1,.0),
 "OTA update in garage":(0,0,.02,.08,.0)}
TEMPL = ["{m} {f} while {s}", "Hazard: {m} the {f} function during {s}",
         "Malfunction in {f}: {m} behaviour when {s}", "{s}: {m} {f} observed"]
NUM = ["speed_kmh","pedestrians","exposure_pct","reaction_s","occupants"]

def _e(p): return 1 if p<.1 else 2 if p<1 else 3 if p<10 else 4

def generate(n=3000, seed=42):
    r = np.random.default_rng(seed); rows=[]
    F,M,S_=list(FUNCS),list(MALS),list(SCEN)
    for _ in range(n):
        f,m,sc = str(r.choice(F)), str(r.choice(M)), str(r.choice(S_))
        slo,shi,elo,ehi,pp = SCEN[sc]
        spd = r.uniform(slo,shi) if shi>slo else 0.0; ped=int(r.random()<pp)
        pct = float(np.exp(r.uniform(np.log(elo),np.log(ehi)))); rt=r.uniform(.5,3); occ=int(r.integers(1,6))
        fb,cb = FUNCS[f]; ms,mc = MALS[m]
        s = fb+ms + (int(spd>=80)-int(spd<=10)+ped if fb>0 else 0)
        c = cb+mc + int(spd>=100) - int(rt>=2)
        s,c,e = int(np.clip(s,0,3)), int(np.clip(c,0,3)), _e(pct)
        if r.random()<.06: s=int(np.clip(s+r.choice([-1,1]),0,3))
        if r.random()<.06: e=int(np.clip(e+r.choice([-1,1]),1,4))
        if r.random()<.06: c=int(np.clip(c+r.choice([-1,1]),0,3))
        rows.append(dict(text=compose(f,m,sc,int(r.integers(4))),function=f,speed_kmh=round(spd,1),
            pedestrians=ped,exposure_pct=round(pct,3),reaction_s=round(rt,2),occupants=occ,S=s,E=e,C=c,ASIL=asil(s,e,c)))
    df=pd.DataFrame(rows)
    for col in ["speed_kmh","reaction_s"]:   # realistic missing values
        df.loc[r.random(n)<.02,col]=np.nan
    return df
def compose(f,m,sc,t=0): return TEMPL[t].format(m=m,f=f,s=sc)
if __name__=="__main__":
    generate().to_csv("data/hara_synthetic.csv",index=False); print("saved data/hara_synthetic.csv")

import pandas as pd, numpy as np, glob, sys
fs=sorted(glob.glob("abl_*.csv")); d=pd.concat([pd.read_csv(f,dtype={'tag':str}) for f in fs])
d["ae"]=(d.est-d.truth).abs()
def mae14(g):
    wm=g.groupby("truth").est.mean(); return np.mean(np.abs(wm.values-wm.index.values))
rows=[]
for (f,v),g in d.groupby(["folder","variant"]):
    rows.append(dict(folder=f,variant=v,MAE14=mae14(g),pcMAE=g.groupby("tag").ae.mean().mean(),n_missing=int(g.est.isna().sum()),subres=int(g.groupby("tag").subres.first().sum()),sigma_med=g.groupby("tag").sigma.first().median()))
R=pd.DataFrame(rows); P=R.pivot(index="variant",columns="folder",values="MAE14")
pd.set_option("display.width",250); print(P.round(4).to_string()); print(R.pivot(index="variant",columns="folder",values="subres").to_string())
print(R.pivot(index="variant",columns="folder",values="sigma_med").round(3).to_string())
R.to_csv("abl_summary.csv",index=False)

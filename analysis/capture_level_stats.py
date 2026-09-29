import os, pandas as pd, numpy as np, glob, warnings
from scipy import stats
import statsmodels.formula.api as smf
warnings.filterwarnings("ignore")
B=os.environ.get("VC3_DATA", "../images/")
S={"A4":"W","Wall":"WW","Fabric":"FA","Wood":"WO"}
def load(ref,code):
    f=glob.glob(B+f"*_{ref}_{code}_VisionCrack3/results/results_perwidth_*.csv")[0]
    d=pd.read_csv(f,encoding="utf-8-sig"); d["ref"]=ref; d["tag"]=d["tag"].astype(str); return d
rows=[];pm=[];rng=np.random.default_rng(0)
for s,c in S.items():
    o,t=load("opaque",c),load("transparent",c)
    out={"surf":s}
    for k,d in (("op",o),("tr",t)):
        wm=d.groupby("truth_mm")["est_mm"].mean()
        out[k+"_MAE14"]=np.mean(np.abs(wm-wm.index))
        pc=d.groupby("tag")["err_mm"].apply(lambda e:np.mean(np.abs(e)))
        out[k+"_pcMAE_mean"]=pc.mean(); out[k+"_CI"]=stats.t.ppf(.975,len(pc)-1)*pc.std(ddof=1)/np.sqrt(len(pc))
        out[k+"_n"]=len(pc); out[k+"_pc"]=pc.values
    # original test: Wilcoxon over 14 widths on |width-mean error|
    ao=(o.groupby("truth_mm")["est_mm"].mean()-sorted(o.truth_mm.unique())).abs()
    at=(t.groupby("truth_mm")["est_mm"].mean()-sorted(t.truth_mm.unique())).abs()
    out["p_wilcoxon14"]=stats.wilcoxon(at.values,ao.values).pvalue
    # capture-level: unpaired (different captures)
    out["p_MWU"]=stats.mannwhitneyu(out["tr_pc"],out["op_pc"],alternative="two-sided").pvalue
    out["p_welch"]=stats.ttest_ind(out["tr_pc"],out["op_pc"],equal_var=False).pvalue
    # cluster bootstrap over captures for dMAE (tr - op, per-capture MAE mean)
    bs=[rng.choice(out["tr_pc"],10).mean()-rng.choice(out["op_pc"],10).mean() for _ in range(20000)]
    out["d"]=out["tr_pc"].mean()-out["op_pc"].mean(); out["d_lo"],out["d_hi"]=np.percentile(bs,[2.5,97.5])
    # mixed model: |err| ~ ref + C(width) + (1|capture)
    d=pd.concat([o,t]); d["ae"]=d.err_mm.abs(); d["cap"]=d.ref+"_"+d.tag; d["w"]=d.truth_mm.astype(str)
    m=smf.mixedlm("ae ~ C(ref, Treatment('opaque')) + C(w)",d,groups=d["cap"]).fit(reml=True)
    key=[k for k in m.params.index if "transparent" in k][0]
    out["lmm_beta"]=m.params[key]; out["lmm_p"]=m.pvalues[key]; out["lmm_capvar"]=float(m.cov_re.iloc[0,0]); out["lmm_resvar"]=m.scale
    rows.append(out)
R=pd.DataFrame(rows)
def holm(p):
    p=np.asarray(p);o=np.argsort(p);adj=np.empty_like(p);mx=0
    for i,j in enumerate(o): mx=max(mx,(len(p)-i)*p[j]); adj[j]=min(1,mx)
    return adj
for k in ("p_wilcoxon14","p_MWU","p_welch","lmm_p"): R[k+"_holm"]=holm(R[k])
pd.set_option("display.width",250); pd.set_option("display.precision",4)
print(R[["surf","op_MAE14","tr_MAE14","op_pcMAE_mean","op_CI","tr_pcMAE_mean","tr_CI","p_wilcoxon14"]])
print(R[["surf","d","d_lo","d_hi","p_MWU","p_MWU_holm","p_welch","p_welch_holm","lmm_beta","lmm_p","lmm_p_holm","p_wilcoxon14_holm"]])
print(R[["surf","lmm_capvar","lmm_resvar"]])
for _,r in R.iterrows(): print(r.surf,"op",np.round(r.op_pc,4),"tr",np.round(r.tr_pc,4))
R.drop(columns=["op_pc","tr_pc"]).to_csv("capture_level_stats.csv",index=False)

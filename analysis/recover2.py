import numpy as np, cv2, pandas as pd, glob, json, os, sys, time, warnings, ast
warnings.filterwarnings("ignore")
import vc6
from runone import run
from find_cr2 import find_rows
from recover import find_cr_fast, approx_from_fig, approx_from_tpl, U, S
def ref_rows(g, x0, x1, lo, hi):
    bcl=255.0-np.median(g[lo:hi+1, x0:x1].astype(float),axis=0)
    bars=vc6.detect_bars_binary_search(bcl); grades=vc6.classify_thickness(bars)
    okb=[(b,gr) for b,gr in zip(bars,grades) if gr in vc6.GRADE_MM]
    ref=vc6.build_reference_profiles(bcl,[b for b,_ in okb],[gr for _,gr in okb])
    sig=vc6.estimate_psf_sigma_esf(ref) or vc6.estimate_psf_sigma(ref)
    fp=[];fm=[]
    for (s,e,c,w),gr in okb:
        f=vc6.measure_fwhm(vc6._detrend(bcl[max(0,int(s)-6):min(len(bcl),int(e)+7)]))
        if f: fp.append(f); fm.append(vc6.GRADE_MM[gr])
    ca,cb=np.polyfit(np.asarray(fp),np.asarray(fm),1)
    return 1.0/ca, sig
def to_bc(x0,x1,lo,hi):
    L=hi-lo; bh=int(L/0.5)
    r=int(max(8,0.35*bh)); BCY=L-r
    assert BCY>=0 and BCY-r<=0 and BCY+r<=bh-1
    return (x0,lo,x1-x0,bh),BCY
def dist(v,t): return abs(round(v[0],4)-t[0])/1e-4+abs(round(v[1],4)-t[1])/1e-4
def ev(g,x0,x1,lo,hi):
    try: return ref_rows(g,x0,x1,lo,hi)
    except Exception: return (np.nan,np.nan)
def search(g,x0,x1,lo0,hi0,t):
    best=[]
    for lo in range(lo0-60,lo0+61,2):
        for hi in range(hi0-60,hi0+61,2):
            if hi-lo<30: continue
            d=dist(ev(g,x0,x1,lo,hi),t)
            if np.isfinite(d): best.append((d,x0,x1,lo,hi))
    best.sort(); cur=best[0]
    if cur[0]==0: return cur
    for it in range(3):
        cands=[]
        for (d,a,b,lo,hi) in best[:3] if it==0 else [cur]:
            for lo2 in range(lo-2,lo+3):
                for hi2 in range(hi-2,hi+3):
                    for a2 in range(a-12,a+13,3):
                        for b2 in range(b-12,b+13,3):
                            dd=dist(ev(g,a2,b2,lo2,hi2),t)
                            if np.isfinite(dd): cands.append((dd,a2,b2,lo2,hi2))
        cands.sort()
        if cands[0][0]==0: return cands[0]
        (d,a,b,lo,hi)=cands[0]; cands2=[]
        for lo2 in range(lo-3,lo+4):
            for hi2 in range(hi-3,hi+4):
                for a2 in range(a-2,a+3):
                    for b2 in range(b-2,b+3):
                        dd=dist(ev(g,a2,b2,lo2,hi2),t)
                        if np.isfinite(dd): cands2.append((dd,a2,b2,lo2,hi2))
        cands2.sort(); cur=min(cands[0],cands2[0])
        if cur[0]==0: return cur
    return cur
def done_from_logs(folder):
    out={}
    for fn in [f"log_{folder}.txt", f"log2_{folder}.txt"]:
        if not os.path.exists(fn): continue
        for l in open(fn):
            if "maxdiff" in l and l.startswith(folder[:14]):
                parts=l.split(" ",2); tag=parts[1]; dct=ast.literal_eval(parts[2][:parts[2].rindex("}")+1])
                if dct.get("maxdiff")==0.0: out[tag]=dct
    return out
def process(folder):
    imgs=sorted(glob.glob(U+folder+"/*.png")); out=done_from_logs(folder); tpl=None
    pw=pd.read_csv(glob.glob(U+folder+"/results/results_perwidth_*.csv")[0],encoding="utf-8-sig",dtype={'tag':str})
    for ip in imgs:
        tag=os.path.splitext(os.path.basename(ip))[0]
        g=cv2.imread(ip,0)
        if tag in out:
            if tpl is None:
                BC=out[tag]["BC"]; tpl=g[BC[1]:BC[1]+BC[3],BC[0]:BC[0]+BC[2]].copy()
            continue
        t0=time.time()
        d=pd.read_csv(glob.glob(U+folder+f"/results/results_profiles_{tag}_*.csv")[0],encoding="utf-8-sig")
        c=(255.0-d[f"{tag}_center_raw"].values).astype(np.uint8); m=255.0-d[f"{tag}_median_raw"].values
        x,Y=find_cr_fast(g,c); rows=find_rows(g,x,Y,len(c),m)
        top,bot=rows[0]; CR=(x,top,len(c),bot-top+1); CRY=Y-top
        ap=approx_from_fig(glob.glob(U+folder+"/results/vc_result_01*.png")[0],g) if tpl is None else approx_from_tpl(tpl,g)
        _,bx,by,bw,bh=ap; r=int(0.35*bh); Yc=by+bh//2
        t=(float(S.loc[tag,'px_per_mm']),float(S.loc[tag,'sigma_px']))
        dbest,x0,x1,lo,hi=search(g,bx,bx+bw,Yc-r,Yc+r,t)
        BC,BCY=to_bc(x0,x1,lo,hi)
        if tpl is None: tpl=g[lo:hi+1,x0:x1].copy()
        px,sg,pairs,_=run(ip,BC,BCY,CR,CRY,True)
        ref=pw[pw.tag==tag].est_mm.values; est=np.array([e for _,e in pairs],float)
        md=float(np.nanmax(np.abs(est-ref))) if len(ref)==len(est) else None
        out[tag]=dict(BC=[int(v) for v in BC],BCY=int(BCY),CR=[int(v) for v in CR],CRY=int(CRY),maxdiff=md,refdist=float(dbest),px=round(px,4),sig=round(sg,4),t=t)
        print(folder[:14],tag,out[tag],round(time.time()-t0,1),flush=True)
        json.dump(out,open(f"roi_{folder}.json","w"),indent=1)
    json.dump(out,open(f"roi_{folder}.json","w"),indent=1)
if __name__=="__main__":
    for f in sys.argv[1:]: process(f)

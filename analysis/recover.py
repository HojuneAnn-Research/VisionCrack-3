import numpy as np, cv2, pandas as pd, glob, json, os, sys, time, warnings
warnings.filterwarnings("ignore")
from refpart import ref_part
from runone import run
from find_cr2 import find_rows
U=os.environ.get("VC3_DATA", "../images/")
S=pd.read_csv(os.path.join(os.environ.get('VC3_DATA', '../images/'), 'results_summary.csv'),encoding='utf-8-sig',dtype={'tag':str}).drop_duplicates('tag',keep='last').set_index('tag')
def find_cr_fast(g, c):
    cw=len(c); H,W=g.shape; key=c[:40]
    for y in range(H):
        row=g[y]; xs=np.nonzero(row[:W-cw+1]==c[0])[0]
        for x in xs:
            if np.array_equal(row[x:x+40],key) and np.array_equal(row[x:x+cw],c): return int(x),y
    return None
def bh_for_r(r):
    b=int(np.ceil(r/0.35))
    while int(0.35*b)!=r: b+=1
    return b
def search_bc(g, bx, bw, Yc0, tpx, tsg, dY=40, r0=113, dr=25, dxs=(0,)):
    for dx0 in dxs:
      for dx1 in dxs:
        for r in sorted(range(r0-dr, r0+dr+1), key=lambda v: abs(v-r0)):
          b=bh_for_r(r); BCY=b//2
          for Yc in sorted(range(Yc0-dY, Yc0+dY+1), key=lambda v: abs(v-Yc0)):
            BC=(bx+dx0, Yc-BCY, bw-dx0+dx1, b)
            try: px,sg,n=ref_part(g,BC,BCY)
            except Exception: continue
            if round(px,4)==tpx and round(sg,4)==tsg: return BC,BCY
    return None
def approx_from_fig(fig_path, g):
    f=cv2.imread(fig_path,cv2.IMREAD_GRAYSCALE); panel=f[111:316,67:778]
    gs=cv2.resize(g,(g.shape[1]//4,g.shape[0]//4),interpolation=cv2.INTER_AREA); best=None
    for bw in range(700,861,10):
        for bh in range(280,371,10):
            t=cv2.resize(panel,(bw//4,bh//4),interpolation=cv2.INTER_AREA)
            r=cv2.matchTemplate(gs,t,cv2.TM_CCOEFF_NORMED); _,mx,_,loc=cv2.minMaxLoc(r)
            if best is None or mx>best[0]: best=(mx,loc[0]*4,loc[1]*4,bw,bh)
    return best
def approx_from_tpl(tpl, g):
    gs=cv2.resize(g,(g.shape[1]//4,g.shape[0]//4),interpolation=cv2.INTER_AREA); best=None
    for sc in np.arange(0.90,1.101,0.02):
        t=cv2.resize(tpl,(int(tpl.shape[1]*sc/4),int(tpl.shape[0]*sc/4)),interpolation=cv2.INTER_AREA)
        r=cv2.matchTemplate(gs,t,cv2.TM_CCOEFF_NORMED); _,mx,_,loc=cv2.minMaxLoc(r)
        if best is None or mx>best[0]: best=(mx,loc[0]*4,loc[1]*4,int(tpl.shape[1]*sc),int(tpl.shape[0]*sc))
    return best
def process(folder):
    imgs=sorted(glob.glob(U+folder+"/*.png")); out={}; tpl=None
    pw=pd.read_csv(glob.glob(U+folder+"/results/results_perwidth_*.csv")[0],encoding="utf-8-sig",dtype={'tag':str})
    for i,ip in enumerate(imgs):
        tag=os.path.splitext(os.path.basename(ip))[0]; t0=time.time()
        g=cv2.imread(ip,cv2.IMREAD_GRAYSCALE)
        pc=glob.glob(U+folder+f"/results/results_profiles_{tag}_*.csv")[0]
        d=pd.read_csv(pc,encoding="utf-8-sig")
        c=(255.0-d[f"{tag}_center_raw"].values).astype(np.uint8); m=255.0-d[f"{tag}_median_raw"].values
        x,Y=find_cr_fast(g,c); rows=find_rows(g,x,Y,len(c),m)
        if len(rows)!=1: out[tag]=dict(err="cr rows %d"%len(rows)); print(tag,out[tag]); continue
        top,bot=rows[0]; CR=(x,top,len(c),bot-top+1); CRY=Y-top
        ap=approx_from_fig(glob.glob(U+folder+"/results/vc_result_01*.png")[0],g) if tpl is None else approx_from_tpl(tpl,g)
        _,bx,by,bw,bh=ap; Yc0=by+bh//2
        tpx,tsg=float(S.loc[tag,'px_per_mm']),float(S.loc[tag,'sigma_px'])
        res=search_bc(g,bx,bw,Yc0,tpx,tsg,r0=int(0.35*bh))
        if res is None: res=search_bc(g,bx,bw,Yc0,tpx,tsg,r0=int(0.35*bh),dxs=(-16,-8,8,16))
        if res is None: out[tag]=dict(err="bc not found",CR=CR,CRY=CRY,approx=[int(v) for v in ap[1:]]); print(tag,out[tag]); continue
        BC,BCY=res
        if tpl is None: tpl=g[BC[1]:BC[1]+BC[3],BC[0]:BC[0]+BC[2]].copy()
        px,sg,pairs,_=run(ip,BC,BCY,CR,CRY,True)
        ref=pw[pw.tag==tag].est_mm.values; est=np.array([e for t,e in pairs],float)
        md=float(np.nanmax(np.abs(est-ref))) if len(ref)==len(est) else None
        out[tag]=dict(BC=[int(v) for v in BC],BCY=int(BCY),CR=[int(v) for v in CR],CRY=int(CRY),maxdiff=md)
        print(folder[:14],tag,out[tag],round(time.time()-t0,1),flush=True)
    return out
if __name__=="__main__":
    folder=sys.argv[1]; o=process(folder)
    json.dump(o,open(f"roi_{folder}.json","w"),indent=1)

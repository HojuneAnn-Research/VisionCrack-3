import numpy as np, pandas as pd, json, glob, os, sys, warnings, io, contextlib
warnings.filterwarnings("ignore")
import vc6, vcmod
U=os.environ.get("VC3_DATA", "../images/")
TRUTH=vc6.TRUTH
def run(img,roi,**kw):
    st={}
    with contextlib.redirect_stdout(io.StringIO()):
        pxmm,sig,res,info=vcmod.measure_mod(img,tuple(roi["BC"]),roi["BCY"],tuple(roi["CR"]),roi["CRY"],stats=st,**kw)
        res,_=vc6.keystone_correct(res)
        if len(res)>len(TRUTH): res=vc6.drop_ghosts_by_pitch(res,len(TRUTH))
    off=len(TRUTH)-len(res); t_use=TRUTH[off:] if off>0 else TRUTH[:len(res)]
    return sig,[(t,round(p,4) if p else None) for t,(s,e,p) in zip(t_use,res)],st
VARIANTS={
 "full":{},
 "no_masked_bg":dict(masked_bg=False),
 "lsf_sigma":dict(esf_sigma=False),
 "no_deconv":dict(deconv=False),
 "rev3_window":dict(band_frac=0.35,win_frac=1.0),
 "margin1.0":dict(margin_k=1.0),"margin2.0":dict(margin_k=2.0),
 "smooth15":dict(smooth=15),"smooth35":dict(smooth=35),
 "band0.60":dict(band_frac=0.60),"win0.4":dict(win_frac=0.4),"win0.8":dict(win_frac=0.8),
}
if __name__=="__main__":
    folder=sys.argv[1]; names=sys.argv[2].split(",") if len(sys.argv)>2 else list(VARIANTS)
    extra=json.loads(sys.argv[3]) if len(sys.argv)>3 else {}
    rf=f"roi_{folder}.json"; rois=json.load(open(rf if os.path.exists(rf) else os.path.join("results",rf))); rows=[]
    for tag,roi in rois.items():
        img=glob.glob(U+folder+f"/{tag}.png")[0]
        for vn in names:
            kw=dict(VARIANTS.get(vn,{})); kw.update(extra.get(vn,{}))
            try: sig,pairs,st=run(img,roi,**kw)
            except Exception as ex: print(tag,vn,"FAIL",ex,flush=True); continue
            for t,e in pairs: rows.append(dict(folder=folder,tag=tag,variant=vn,truth=t,est=e,sigma=sig,subres=st.get('subres',0),n=st.get('n',0)))
        print(folder[:14],tag,"done",flush=True)
    out=sys.argv[4] if len(sys.argv)>4 else f"abl_{folder}.csv"
    pd.DataFrame(rows).to_csv(out,index=False)

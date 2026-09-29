import numpy as np, warnings; warnings.filterwarnings("ignore")
import vc6
TRUTH=vc6.TRUTH
def run(img, BC, BCY, CR, CRY, keystone, **kw):
    pxmm,sig,res,info=vc6.measure_v2(img,BC,BCY,CR,CRY,**kw)
    if keystone: res,_=vc6.keystone_correct(res)
    if len(res)>len(TRUTH): res=vc6.drop_ghosts_by_pitch(res,len(TRUTH))
    off=len(TRUTH)-len(res); t_use=TRUTH[off:] if off>0 else TRUTH[:len(res)]
    est=[round(p,4) if p else None for (s,e,p) in res]
    return pxmm,sig,list(zip(t_use,est)),info

import numpy as np
import vc6
def ref_part(g, BC, BCY, esf_sigma=True):
    bx,by,bw,bh=BC; bc=g[by:by+bh, bx:bx+bw]
    y0,y1=vc6._band(bh,BCY)
    bcl=255.0-np.median(bc[y0:y1+1,:].astype(float),axis=0)
    bars=vc6.detect_bars_binary_search(bcl); grades=vc6.classify_thickness(bars)
    okb=[(b,gr) for b,gr in zip(bars,grades) if gr in vc6.GRADE_MM]
    ref=vc6.build_reference_profiles(bcl,[b for b,_ in okb],[gr for _,gr in okb])
    sig=(vc6.estimate_psf_sigma_esf(ref) if esf_sigma else None) or vc6.estimate_psf_sigma(ref)
    FWpsf=2.355*sig; fw_px=[];fw_mm=[]
    for (s,e,c,w),gr in okb:
        f=vc6.measure_fwhm(vc6._detrend(bcl[max(0,int(s)-6):min(len(bcl),int(e)+7)]))
        if f: fw_px.append(f); fw_mm.append(vc6.GRADE_MM[gr])
    ca,cb=np.polyfit(np.asarray(fw_px),np.asarray(fw_mm),1)
    return 1.0/ca, sig, len(bars)

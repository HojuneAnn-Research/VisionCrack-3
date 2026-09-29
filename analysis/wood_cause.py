"""Target-side analysis of the opaque vs transparent difference (Experiment 3, "Tests of the cause").
Outputs: wood_cause_lines.csv (per capture and gauge line), wood_cause_texture.csv (texture correlations),
and a printed summary. Run from analysis/; data are read from ../images/ or $VC3_DATA."""
import os, io, json, contextlib, itertools, numpy as np, pandas as pd
import vc6, vcmod
from vc6 import _band, _detrend, _masked_background, measure_fwhm, load_gray
U = os.environ.get("VC3_DATA", "../images/"); TR = vc6.TRUTH
def roi(f):
    rf = f"roi_{f}_VisionCrack3.json"
    return json.load(open(rf if os.path.exists(rf) else os.path.join("results", rf)))
def run(f, tag, r):
    img = os.path.join(U, f + "_VisionCrack3", tag + ".png")
    with contextlib.redirect_stdout(io.StringIO()):
        pxmm, sig, res, info = vcmod.measure_mod(img, tuple(r["BC"]), r["BCY"], tuple(r["CR"]), r["CRY"])
    g = load_gray(img); cx, cy, cw, ch = r["CR"]; segs = info["segs"]; FW = 2.355 * sig
    y0, y1 = _band(ch, r["CRY"], 0.85); rows = g[cy:cy+ch, cx:cx+cw][y0:y1+1, :].astype(float)
    raw = 255 - np.median(rows, axis=0); bg = _masked_background(raw, segs, int(round(1.5 * FW)), smooth=25)
    lines = [(255 - q) - bg for q in rows]; fw = {}
    for (s, e, c, w) in segs:   # FWHM before any reference-based correction (same window as the measurement)
        m = max(6, int(0.6 * (int(e) - int(s)))); a, b = max(0, int(s) - m), min(len(raw) - 1, int(e) + m)
        v = [measure_fwhm(_detrend(np.asarray(l)[a:b+1])) for l in lines]; v = [x for x in v if x]
        fw[(s, e)] = float(np.median(v)) if v else np.nan
    ndet = len(res); res2, _ = vc6.keystone_correct(res)
    if len(res2) > len(TR): res2 = vc6.drop_ghosts_by_pitch(res2, len(TR))
    off = len(TR) - len(res2); t_use = TR[off:] if off > 0 else TR[:len(res2)]
    out = [dict(cond=f, tag=tag, truth=t, est=p, fwhm_mm=fw[(s, e)] / pxmm, ndet=ndet) for t, (s, e, p) in zip(t_use, res2)]
    # background texture between the lines, in gauge coordinates (line index), lines masked
    tex = None
    if len(res2) == 14:
        cen = np.array([(s + e) / 2 for s, e, p in res2]); rawI = np.median(rows, axis=0)
        u = np.interp(np.arange(len(rawI)), cen, np.arange(14), left=np.nan, right=np.nan); ok = ~np.isnan(u)
        grid = np.linspace(0, 13, 1301); v = np.interp(grid, u[ok], rawI[ok])
        v = v - np.convolve(v, np.ones(41) / 41, "same"); tex = np.where(np.abs(grid - np.round(grid)) > 0.3, v, np.nan)
    return out, tex
def mae14(q):
    wm = q.groupby("truth").est.mean(); return float(np.mean(np.abs(wm - wm.index)))
def corr(a, b):
    m = ~np.isnan(a) & ~np.isnan(b); return float(np.corrcoef(a[m], b[m])[0, 1])
PAIRS = [("4_opaque_WO", "8_transparent_WO"), ("3_opaque_FA", "7_transparent_FA"), ("2_opaque_WW", "6_transparent_WW")]
rows, texrows, TEX = [], [], {}
for pair in PAIRS:
    for f in pair:
        for tag, r in sorted(roi(f).items()):
            o, t = run(f, tag, r); rows += o
            if t is not None: TEX[(f, tag)] = t
    for a, b in itertools.combinations([k for k in TEX if k[0] in pair], 2):
        texrows.append(dict(pair=pair[0][2:] + "/" + pair[1][2:], kind="within " + a[0] if a[0] == b[0] else "between", r=corr(TEX[a], TEX[b])))
d = pd.DataFrame(rows); d["err"] = d.est - d.truth; d.to_csv("wood_cause_lines.csv", index=False)
T = pd.DataFrame(texrows); T.to_csv("wood_cause_texture.csv", index=False)
o, t = d[d.cond == "4_opaque_WO"], d[d.cond == "8_transparent_WO"]
print("MAE14 opaque / transparent wood:", round(mae14(o), 3), round(mae14(t), 3))
print("0.85 mm line error, opaque wood (per capture):", o[o.truth == 0.85].err.round(2).tolist())
print("opaque MAE14 without 0.85 mm line:", round(mae14(o[o.truth != 0.85]), 3))
print("0.50-0.75 mm lines MAE14 opaque / transparent:", round(mae14(o[o.truth <= 0.75]), 3), round(mae14(t[t.truth <= 0.75]), 3))
print("0.50-0.75 mm mean error, opaque:", o[o.truth <= 0.75].groupby("truth").err.mean().round(3).to_dict())
print(">=0.90 mm lines MAE14 opaque / transparent:", round(mae14(o[o.truth >= 0.9]), 3), round(mae14(t[t.truth >= 0.9]), 3))
print("FWHM before correction, 0.50 mm line, opaque / transparent:", round(o[o.truth == 0.5].fwhm_mm.mean(), 2), round(t[t.truth == 0.5].fwhm_mm.mean(), 2))
bs = os.path.join("results", "bothswap_4.csv") if not os.path.exists("bothswap_4.csv") else "bothswap_4.csv"
if os.path.exists(bs): print("both constants exchanged, opaque wood MAE14:", round(mae14(pd.read_csv(bs)), 3))
print(T.groupby(["pair", "kind"]).r.mean().round(2).to_string())

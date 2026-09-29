from vc6 import *
from vc6 import _band,_detrend,_plateau,_rect_gauss,_fit_width_fixedA,_detect_crack_segments,_masked_background
import numpy as np
from scipy.optimize import curve_fit
def measure_mod(img, BC, BCY, CR, CRY, nrows=10, masked_bg=True, esf_sigma=True,
               band_frac=0.85, win_frac=0.6, sig_override=None, kq_factor=None, deconv=True, margin_k=1.5, smooth=25, stats=None):
    """v1 measure()와 동일 인터페이스/반환형. masked_bg=False → v1과 수치 동일.
    (nrows는 v1 호환용으로만 유지 — v1과 동일하게 실제로는 사용되지 않으며,
     스캔 밴드의 전체 행이 행별 FWHM→median 집계에 쓰인다.)

    [rev4 변경점] 두 개의 스칼라만 추가. 근거는 직조(우븐) 배경 실측에서 나온
    두 층위의 오차원인.
      band_frac (기존 하드코딩 0.35 → 0.85): 측정/검출에 쓰는 스캔 밴드를 막대
        전체 높이로 확대. 위브 weft(가로실)가 행마다 프로파일 어깨를 다르게
        들어올려 FWHM을 흔드는데(단발 밴드에서 CRY ±15px에 MAE 0.084↔0.185),
        전체 높이 행별-median으로 그 변조를 상쇄. 실측 MAE 0.185→0.106.
      win_frac (기존 창폭 m=max(8, e-s) → max(6, 0.6·(e-s))): FWHM 측정창을
        세그폭의 ±100%→±60%로 축소. 막대 옆에 warp(세로실)가 붙으면 넓은 창이
        그 어깨를 반치 교차에 포함해 폭을 과대측정(1.0mm FWHM 24.4→17.9px로 복귀,
        넓은 막대 2.0/2.5mm는 불변). warp는 막대와 같은 fx축이라 FFT 노치로는
        분리 불가 → 공간영역(창 축소)만이 유효.

    기각 기록(재시도 방지): 2D FFT 위브 노치. weft(fy축) 노치는 무효(전체높이
    적분과 중복), warp(fx축) 노치는 막대를 함께 지워 MAE 4.2로 전멸. isotropic
    링노치는 특정 막대는 잡아도 고조파 손상으로 설정 민감(MAE 0.17~2.8).
    band_frac=0.35, win_frac=1.0 으로 rev3 동작 복귀 가능."""
    g = load_gray(img)

    # ── 참조(바코드): v1과 완전 동일 ──
    bx, by, bw, bh = BC
    bc = g[by:by + bh, bx:bx + bw]
    y0, y1 = _band(bh, BCY)
    bcl = 255.0 - np.median(bc[y0:y1 + 1, :].astype(float), axis=0)
    bars = detect_bars_binary_search(bcl)
    grades = classify_thickness(bars)
    okb = [(b, gr) for b, gr in zip(bars, grades) if gr in GRADE_MM]
    ref = build_reference_profiles(bcl, [b for b, _ in okb], [gr for _, gr in okb])
    sig = (estimate_psf_sigma_esf(ref) if esf_sigma else None) or estimate_psf_sigma(ref)
    if sig_override is not None: sig = float(sig_override)
    FWpsf = 2.355 * sig
    pxthr = 1.4 * FWpsf

    fw_px, fw_mm = [], []
    for (s, e, c, w), gr in okb:
        f = measure_fwhm(_detrend(bcl[max(0, int(s) - 6):min(len(bcl), int(e) + 7)]))
        if f:
            fw_px.append(f); fw_mm.append(GRADE_MM[gr])
    fw_px = np.asarray(fw_px, float); fw_mm = np.asarray(fw_mm, float)
    ca, cb = np.polyfit(fw_px, fw_mm, 1)
    msk = fw_px > FWpsf * 1.05
    kq = float(np.median(np.sqrt(fw_px[msk] ** 2 - FWpsf ** 2) / fw_mm[msk]))
    pxmm = 1.0 / ca
    if stats is not None: stats['kqf'] = kq / pxmm
    if kq_factor is not None: kq = float(kq_factor) * pxmm

    def fwcal(f):
        if f is None:
            return None
        if (not deconv) or f < pxthr:
            return float(ca * f + cb)
        return float(np.sqrt(max(f * f - FWpsf ** 2, 0.0)) / kq)

    # fixed-A 보정(참조): v1과 동일
    Pb = _plateau(bcl - bcl.min(), bars)
    sa = sb = None
    if Pb:
        bwp, bm = [], []
        for (s, e, c, w), gr in okb:
            a2, b2 = max(0, int(s) - 8), min(len(bcl), int(e) + 8)
            yb = bcl[a2:b2] - bcl[a2:b2].min()
            xb = np.arange(a2, b2, dtype=float)
            try:
                p, _ = curve_fit(lambda x, c_, w_, b_: _rect_gauss(x, c_, w_, b_, Pb, sig),
                                 xb, yb, p0=[(s + e) / 2, e - s, 0],
                                 bounds=([a2, 0.1, -30], [b2, 130, 30]), maxfev=8000)
                bwp.append(p[1]); bm.append(GRADE_MM[gr])
            except Exception:
                pass
        if len(bwp) >= 2:
            sa, sb = np.polyfit(np.asarray(bwp, float), np.asarray(bm, float), 1)
    SUBRES_MM = pxthr / pxmm

    # ── 타깃 검출: v1 그대로 (median-filter 배경, 지역대비 정규화) ──
    segs, crl2, cr_lines2 = _detect_crack_segments(g, CR, CRY, pxmm, band_frac)

    # ── [MASKED-BG] 측정용 프로파일 재구성 ──
    if masked_bg:
        cx, cy, cw, ch = CR
        ry0, ry1 = _band(ch, CRY, band_frac)
        rows = g[cy:cy + ch, cx:cx + cw][ry0:ry1 + 1, :].astype(float)
        raw_band = 255.0 - np.median(rows, axis=0)
        margin = int(round(margin_k * FWpsf))
        bg = _masked_background(raw_band, segs, margin, smooth=smooth)
        crl2 = raw_band - bg
        cr_lines2 = [(255.0 - r) - bg for r in rows]

    P = _plateau(np.asarray(crl2), segs)

    # ── 폭 측정: rev3와 동일 로직, 창폭만 win_frac로 축소 ──
    res = []
    for (s, e, c, w) in segs:
        m = max(6, int(win_frac * (int(e) - int(s))))
        s_p, e_p = max(0, int(s) - m), min(len(crl2) - 1, int(e) + m)
        fws = []
        for l in cr_lines2:
            f = measure_fwhm(_detrend(np.asarray(l)[s_p:e_p + 1]))
            if f:
                fws.append(f)
        fw = float(np.median(fws)) if fws else None
        mm = fwcal(fw)
        if mm is not None and sa is not None and P and mm < SUBRES_MM:
            wp = _fit_width_fixedA(np.asarray(crl2), int(s), int(e), sig, P)
            if wp is not None:
                mm = float(sa * wp + sb)
                if stats is not None: stats['subres'] = stats.get('subres',0)+1
        if stats is not None: stats['n'] = stats.get('n',0)+1
        res.append((s, e, mm))

    cx, cy, cw, ch = CR
    info = dict(bc=bc, cr=g[cy:cy + ch, cx:cx + cw], BCY=BCY, CRY=CRY,
                bars=bars, grades=grades, bcl=bcl, crl2=crl2, segs=segs,
                masked_bg=masked_bg, sigma=sig, FWpsf=FWpsf, pxmm=pxmm)
    return pxmm, sig, res, info



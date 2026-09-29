# -*- coding: utf-8 -*-
"""
rev2.py — VisionCrack 투명 바코드 측정 파이프라인 (개선판, 단일 파일 독립 실행)
================================================================================
v1(vc_measure.measure) 대비 변경점은 단 하나:

  [MASKED-BG] 타깃 '측정' 단계의 배경 추정을
      median filter  →  "검출된 선 구간 마스킹 후 선형 보간"
  으로 교체. 선 '검출'은 기존 median-filter 경로를 그대로 유지(강건성 보존).

  근거: v1의 median filter(창 ~130px)는 넓은 선(≥1.5mm)의 창 점유율 때문에
  배경을 위로 끌어올려 선 진폭을 깎는다. 이 하나가 세 가지 관측 오차의 공통
  원인이었다 — (1) 2.5mm FWHM 잘림, (2) plateau P 과소추정→fixed-A의
  sub-res(0.1–0.45mm) 전 구간 +0.1mm대 편향, (3) 0.85–1.5mm 완만한 +드리프트.

  전체 사진(고해상) 4배경 게이지 A/B (MAE, mm):
      White  0.0276→0.0215 (−22%)   BG1 0.0736→0.0998 (+36%*)
      BG2    0.1021→0.0763 (−25%)   BG3 0.0397→0.0292 (−26%)
  * BG1은 필름 들뜸/정반사로 선 자체가 PSF 이상으로 번진 촬영측 문제로 진단.
    masked_bg=False 로 v1 동작 복귀 가능(수치까지 동일함을 검증).

  기각한 접근(재시도 방지 기록):
    - PSF(rect⊗Gauss) 단일 측정기로 전 구간 통일: 전 배경 2~4배 악화
      (White .072 / BG1 .128 / BG2 .200 / BG3 .141; +mask도 열세).
      투명 필름은 바 아래 배경이 비쳐 plateau가 변조되는데, A 고정 LSQ는
      평탄부 샘플까지 잔차에 넣어 이를 폭 오차로 흡수. FWHM은 반치 교차점
      2개만 쓰는 국소 측정이라 강건. → 하이브리드 유지가 실증적으로 옳음:
      해상 구간(FWHM 존재)=FWHM+경험 보정, 서브해상(FWHM이 FWHM_psf로
      수렴해 정보 상실)=peak-deficit 피팅.
    - exact FWHM↔w 모델 역변환: 경험적 kq가 프린트 게인·모델 오차까지 흡수해 우세.
    - multi-band 앙상블: _band(frac=0.35)가 이미 바 높이 ~70%를 집계.
    - target-side σ 재추정: 텍스처가 gradient LSF를 부풀려 전 배경 악화.
    - 2단계 잔차 affine: 무효. 참조 대칭 배경제거: 바 점유율 50%라 붕괴.

사용:
    from rev2 import measure_crack, measure_v2, drop_ghosts_by_pitch

    # (A) 실사용 — 균열 하나, 값 하나:
    width_mm, detail = measure_crack(img_path, BC, BCY, CR, CRY)

    # (B) 검증/게이지 — 영역 내 모든 선:
    pxmm, sig, res, info = measure_v2(img_path, BC, BCY, CR, CRY)
    # res = [(seg_start, seg_end, width_mm|None), ...]

    # BC/CR = (x, y, w, h),  BCY/CRY = 영역 내 스캔 중심 y(픽셀)
"""
import sys
import numpy as np, cv2
from scipy.ndimage import median_filter, maximum_filter1d, uniform_filter1d
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
from scipy.special import erf

PDF_DPI = 2400

GRADE_MM = {'T1': 0.5, 'T2': 1.0, 'T3': 1.5, 'T4': 2.0, 'T5': 2.5}
GRADE_KEYS = ('T1', 'T2', 'T3', 'T4', 'T5')
TRUTH = [0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0,1.5,2.0,2.5]


def _smooth(arr, hw):
    """Simple box-car smoothing."""
    k = np.ones(2*hw+1) / (2*hw+1)
    return np.convolve(arr, k, mode='same')


def _detrend(segment):
    """세그먼트 양끝 평균을 잇는 선형 베이스라인 제거 후 clip(0)."""
    seg = np.array(segment, dtype=float)
    n = len(seg)
    if n < 2:
        return seg
    k = max(1, n // 20)
    y0 = float(np.mean(seg[:k]))
    y1 = float(np.mean(seg[-k:]))
    baseline = np.linspace(y0, y1, n)
    return np.clip(seg - baseline, 0, None)


def _normalize(arr):
    """0~1 정규화."""
    a = np.array(arr, dtype=float)
    lo, hi = a.min(), a.max()
    return (a - lo) / (hi - lo + 1e-6)


def _segs_at_thresh(line_inv, mid, min_width=2):
    """임계 mid에서 연속 구간(바) 추출."""
    binary = (line_inv > mid).astype(np.uint8)
    segs = []
    in_bar = False
    start = 0
    for i, v in enumerate(binary):
        if v and not in_bar:
            in_bar = True
            start = i
        elif not v and in_bar:
            in_bar = False
            w = i - start
            if w >= min_width:
                segs.append((start, i-1, (start+i)//2, w))
    if in_bar:
        w = len(binary) - start
        if w >= min_width:
            segs.append((start, len(binary)-1, (start+len(binary))//2, w))
    return segs


def detect_bars_binary_search(line_inv, min_width=1, target=18):
    """이진탐색으로 정확히 target개 바가 잡히는 임계를 찾음."""
    lo = float(line_inv.min())
    hi = float(line_inv.max())
    for _ in range(64):
        mid = (lo + hi) / 2.0
        segs = _segs_at_thresh(line_inv, mid, min_width)
        n = len(segs)
        if n == target:
            return segs
        if n > target:
            lo = mid
        else:
            hi = mid
    best, best_segs = None, []
    for mid in np.linspace(line_inv.min(), line_inv.max(), 500):
        segs = _segs_at_thresh(line_inv, mid, min_width)
        if best is None or abs(len(segs) - target) < abs(best - target):
            best, best_segs = len(segs), segs
        if best == target:
            break
    print(f"  Warning: found {best} bars (target {target})")
    return best_segs


def measure_fwhm(segment):
    """FWHM(px) — 반치폭, 교차점 선형보간. 실패 시 None."""
    baseline = float(np.min(segment))
    peak = float(np.max(segment))
    if peak - baseline < 1.0:
        return None
    half = baseline + (peak - baseline) * 0.5
    left = None
    for i in range(len(segment)):
        if segment[i] >= half:
            if i == 0:
                left = 0.0
            else:
                left = (i-1) + (half - segment[i-1]) / (segment[i] - segment[i-1] + 1e-12)
            break
    right = None
    for i in range(len(segment)-1, -1, -1):
        if segment[i] >= half:
            if i == len(segment)-1:
                right = float(len(segment)-1)
            else:
                right = i + (half - segment[i]) / (segment[i+1] - segment[i] + 1e-12)
            break
    if left is None or right is None or right <= left:
        return None
    return right - left


def _merge_overlapping_segs(segs, gap_tol):
    """겹치거나 맞닿은 세그먼트를 union으로 병합."""
    if not segs:
        return []
    segs = sorted(segs, key=lambda x: x[0])
    merged = [list(segs[0])]
    for s, e, c, w in segs[1:]:
        ls, le, lc, lw = merged[-1]
        if s <= le + gap_tol:
            ne = max(le, e)
            merged[-1] = [ls, ne, (ls+ne)//2, ne-ls]
        else:
            merged.append([s, e, c, w])
    return [tuple(m) for m in merged]


def classify_thickness_gap(bars, n_grades=5):
    """Gap-based 분류 (validation에서도 재사용)."""
    if not bars:
        return []
    widths = np.array([w for (_, _, _, w) in bars], dtype=np.float32)
    indices = np.argsort(widths)
    if len(bars) < n_grades:
        result = [''] * len(bars)
        for rank, idx in enumerate(indices):
            result[idx] = f'T{rank+1}'
        return result
    sorted_widths = widths[indices]
    gaps = np.diff(sorted_widths)
    largest_gap_indices = np.argsort(-gaps)[:n_grades-1]
    largest_gap_indices = np.sort(largest_gap_indices)
    boundaries = sorted_widths[largest_gap_indices + 1]
    result = [''] * len(bars)
    for orig_idx, w in enumerate(widths):
        grade = 1
        for boundary in boundaries:
            if w >= boundary:
                grade += 1
        result[orig_idx] = f'T{grade}'
    return result


def validate_classification(bars, grades, n_grades=5, n_iterations=2):
    """등급 내 폭 일관성 검증 → 이상치(mean+2σ 초과) 제거 후 재분류."""
    widths = np.array([w for (_, _, _, w) in bars], dtype=np.float32)
    result_grades = list(grades)
    for iteration in range(n_iterations):
        grade_widths = {}
        for i, g in enumerate(result_grades):
            if g not in grade_widths:
                grade_widths[g] = []
            grade_widths[g].append(widths[i])
        outliers = set()
        for g, w_list in grade_widths.items():
            if len(w_list) < 2:
                continue
            w_arr = np.array(w_list)
            std = float(np.std(w_arr))
            mean = float(np.mean(w_arr))
            threshold = mean + 2 * std
            for i, (b, grade) in enumerate(zip(bars, result_grades)):
                _, _, _, w = b
                if grade == g and w > threshold:
                    outliers.add(i)
        if not outliers:
            return result_grades
        valid_bars = [b for i, b in enumerate(bars) if i not in outliers]
        new_grades = classify_thickness_gap(valid_bars, n_grades=n_grades)
        new_result = [''] * len(bars)
        valid_idx = 0
        for i in range(len(bars)):
            if i in outliers:
                new_result[i] = '?'
            else:
                new_result[i] = new_grades[valid_idx]
                valid_idx += 1
        result_grades = new_result
    return result_grades


def classify_thickness(bars, n_grades=5):
    """바 너비 → 등급(T1~T5): gap clustering + 검증."""
    result = classify_thickness_gap(bars, n_grades=n_grades)
    if not bars or len(bars) < n_grades:
        return result
    return validate_classification(bars, result, n_grades=n_grades)


def build_reference_profiles(bc_line, bars, grades, edge=5):
    """등급별 바 프로파일 정렬/리샘플/메도이드 평균 → 기준 프로파일."""
    groups = {g: [] for g in GRADE_KEYS}
    for (s, e, c, w), g in zip(bars, grades):
        s_p = max(0, s - edge)
        e_p = min(len(bc_line) - 1, e + edge)
        x_rel = np.arange(s_p, e_p + 1) - c
        groups[g].append((x_rel, bc_line[s_p:e_p+1].copy(), w))
    result = {}
    for g, items in groups.items():
        if not items:
            continue
        lengths = [len(y) for (_, y, _) in items]
        target_len = int(np.median(lengths))
        resampled = []
        for (_, y, _) in items:
            resampled.append(np.interp(np.linspace(0, 1, target_len),
                                       np.linspace(0, 1, len(y)), y))
        if len(resampled) == 1:
            raw_y = np.array(resampled[0])
        else:
            stacked = np.array(resampled)
            dists = ((stacked[:, None] - stacked[None, :])**2).sum(axis=2)
            medoid_idx = int(np.argmin(dists.sum(axis=1)))
            raw_y = stacked[medoid_idx]
        win = max(5, int(target_len * 0.2) | 1)
        poly = 3
        if target_len > win:
            mean_y = savgol_filter(raw_y, win, poly)
            mean_y = np.clip(mean_y, 0, None)
        else:
            mean_y = raw_y
        mean_x = np.linspace(-target_len // 2, target_len // 2, target_len)
        result[g] = {
            'profiles_rel': [(xr, y) for (xr, y, _) in items],
            'profiles_resampled': resampled,
            'mean_x': mean_x,
            'mean_y': mean_y,
            'avg_px': float(np.mean([w for (_, _, w) in items])),
        }
    return result


def estimate_psf_sigma(ref):
    """기준 프로파일의 LSF(=|gradient|) 가우시안 적합 → PSF σ(px)."""
    def _gauss(x, x0, sigma, A):
        return A * np.exp(-0.5 * ((x - x0) / (sigma + 1e-9))**2)
    sigmas = []
    for g in ('T2', 'T3', 'T4'):
        if g not in ref:
            continue
        profile = _normalize(np.array(ref[g]['mean_y'], dtype=float))
        lsf = np.abs(np.gradient(profile))
        n = len(lsf)
        mid = n // 2
        for region, offset in [(lsf[:mid], 0), (lsf[mid:], mid)]:
            pk = int(np.argmax(region)) + offset
            span = max(4, int(n * 0.12))
            i0 = max(0, pk - span)
            i1 = min(n, pk + span + 1)
            x = np.arange(i0, i1, dtype=float)
            y = lsf[i0:i1]
            if len(x) < 4 or y.max() < 0.0001:
                continue
            try:
                popt, _ = curve_fit(_gauss, x, y, p0=[float(pk), 2.0, float(y.max())],
                                    bounds=([i0, 0.3, 0], [i1, 8.0, 2.0]), maxfev=2000)
                sigmas.append(abs(float(popt[1])))
            except Exception:
                continue
    sigma = float(np.median(sigmas)) if sigmas else 1.5
    print(f"  PSF sigma = {sigma:.3f} px  (from {len(sigmas)} edge fits)")
    return sigma


# ════════════════════════════════════════════════════════════════════
#  측정 모델 헬퍼 (rect⊗Gaussian)
# ════════════════════════════════════════════════════════════════════


def estimate_psf_sigma_esf(ref):
    """[개선] ESF(에지 확산 함수)에 erf 직접 피팅으로 PSF σ 추정.
    기존 LSF(=|gradient|) 방식은 미분이 텍스처 노이즈를 증폭해 σ를 부풀리고
    (예: 스터코 표면 σ 2.25 vs 실제 ~1.7), 부푼 σ는 서브해상(fixed-A) 폭에
    거의 정비례로 전가된다(좁은 폭 극한에서 w ∝ peak·σ/A). ESF-erf 피팅은
    미분 없이 원 프로파일에 적합하므로 텍스처에 강건하다.
    실측: BG2 서브해상 MAE 0.143→0.072, 전체 0.076→0.051; 타 배경 회귀 없음.
    피팅 실패 시 None 반환(호출측에서 LSF로 폴백)."""
    from scipy.special import erf as _erf
    sigs = []
    for g in ("T2", "T3", "T4", "T5"):
        if g not in ref:
            continue
        y = np.asarray(ref[g]["mean_y"], float)
        n = len(y)
        if n < 16:
            continue
        lo = np.median(np.sort(y)[:max(3, n // 6)])
        hi = np.median(np.sort(y)[-max(3, n // 6):])
        if hi - lo < 20:
            continue
        yn = (y - lo) / (hi - lo)
        iL = int(np.argmax(yn > 0.5))
        iR = n - 1 - int(np.argmax(yn[::-1] > 0.5))
        for i0, sgn in ((iL, +1), (iR, -1)):
            a, b = max(0, i0 - 8), min(n, i0 + 9)
            xw = np.arange(a, b, dtype=float)
            yw = yn[a:b]
            try:
                p, _ = curve_fit(
                    lambda x, x0, s: 0.5 * (1 + sgn * _erf((x - x0) / (np.sqrt(2) * abs(s) + 1e-9))),
                    xw, yw, p0=[i0, 1.8], maxfev=4000)
                s_ = abs(p[1])
                if 0.4 < s_ < 6:
                    sigs.append(s_)
            except Exception:
                pass
    return float(np.median(sigs)) if len(sigs) >= 3 else None


def _rect_gauss(x, c, w, b, A, s):
    """폭 w인 사각선 ⊗ 가우시안 PSF(σ=s). A=잉크 대비(고정 가능), b=베이스라인."""
    z = s * np.sqrt(2.0)
    return b + A * 0.5 * (erf((x - c + w/2)/z) - erf((x - c - w/2)/z))


def _fit_width_fixedA(prof, s, e, sig, A):
    """진폭 A를 잉크대비로 고정하고 폭(px)만 적합 → peak deficit로 sub해상도 선 복원."""
    m = max(10, e - s); a = max(0, s - m); bb = min(len(prof) - 1, e + m)
    x = np.arange(a, bb + 1).astype(float); y = prof[a:bb + 1]; c0 = (s + e) / 2.0
    try:
        p, _ = curve_fit(lambda x, c, w, b: _rect_gauss(x, c, w, b, A, sig),
                         x, y, p0=[c0, max(2.0, e - s), 0.0],
                         bounds=([c0-15, 0.1, -30], [c0+15, 130, 30]), maxfev=8000)
        return float(p[1])
    except Exception:
        return None


def _plateau(prof, segs, minw=20):
    """가장 넓은 바들의 평탄부 대비 → 잉크 대비 A 추정."""
    tops = [np.median(np.sort(prof[s:e+1])[-6:]) for (s, e, *_) in segs if (e - s) >= minw]
    return float(np.median(tops)) if tops else None


# ════════════════════════════════════════════════════════════════════
#  GUI 선택 (좌표 미지정 시) — cv2 기반 간단 버전
# ════════════════════════════════════════════════════════════════════
GUI_MAX = 1000   # GUI 표시 최대 변(px). 큰 사진은 이 크기로 축소해 보여줌.


def load_gray(path):
    """PNG/JPG는 cv2, PDF는 pdf2image로 렌더 → 그레이스케일."""
    if str(path).lower().endswith('.pdf'):
        import pdf2image
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = None
        pages = pdf2image.convert_from_path(path, dpi=PDF_DPI, fmt='ppm')
        rgb = np.array(pages[0])
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)


def _band(h, cy, frac=0.35):
    """중심 cy 기준 ROI 높이의 ±frac 밴드 → 행 많이 평균(세로선이라 폭 안흐려짐, SNR↑)."""
    r = int(max(8, frac * h))
    return max(0, cy - r), min(h - 1, cy + r)


def _detect_crack_segments(g, CR, CRY, pxmm, band_frac=0.85):
    """크랙(자) ROI에서 선 검출 — 지역대비 정규화(배경 변화에 강함).
    반환: (segs, crl2, cr_lines2)."""
    cx, cy, cw, ch = CR
    b0, b1 = _band(ch, CRY, band_frac)
    rows = g[cy:cy+ch, cx:cx+cw][b0:b1+1, :].astype(float)
    crl = 255.0 - np.median(rows, axis=0)
    cr_lines = [255.0 - r for r in rows]
    ks = int(max(31, round(2.5*pxmm*3))); ks += (ks % 2 == 0)
    bg = median_filter(crl.astype(np.float32), size=ks, mode='reflect')
    crl2 = crl - bg
    cr_lines2 = [l - bg for l in cr_lines]
    crl2c = np.clip(crl2, 0, None)
    sm = _smooth(crl2c, hw=max(1, int(0.03*pxmm)))
    peak = float(sm.max())
    win = int(round(1.6*pxmm)); win += (win % 2 == 0)
    env = uniform_filter1d(maximum_filter1d(sm, size=win, mode='reflect'),
                           size=win, mode='reflect')
    noise = 1.4826 * float(np.median(np.abs(sm - np.median(sm))))
    env = np.maximum(env, max(0.08*peak, 6*noise))
    z = sm / (env + 1e-6)
    binary = z > 0.35
    nfloor = max(4*noise, 0.15*peak)   # 표면 텍스처 노이즈 blip 억제 (실선은 ≥0.4·peak)
    raw = []
    i, n = 0, len(sm)
    while i < n:
        if binary[i]:
            s0 = i
            while i < n and binary[i]: i += 1
            e0 = i - 1
            if (e0 - s0) >= 2 and float(crl2c[s0:e0+1].max()) >= nfloor:
                raw.append((s0, e0, (s0+e0)//2, e0-s0))
        else:
            i += 1
    segs = _merge_overlapping_segs(raw, gap_tol=max(2, int(0.3*pxmm)))
    return segs, crl2, cr_lines2


def keystone_correct(res, min_lines=6, cv_max=0.05):
    """등간격 게이지 전용: 검출 선들의 국소 피치로 원근(키스톤) 왜곡 보정.

    자(ruler)의 선은 물리적으로 등간격 → 픽셀 피치가 위치에 따라 변하면 그건
    카메라가 수직이 아니라 생긴 원근 확대/축소다. 각 선을 '국소 피치/평균 피치'로
    나눠 위치별 배율을 제거한다. 등간격이 아니면(실제 크랙 등) 보정 안 함.
    반환: (보정된 res, 적용여부)."""
    pts = [((s+e)/2.0, p, i) for i, (s, e, p) in enumerate(res) if p is not None]
    if len(pts) < min_lines:
        return res, False
    centers = np.array([c for c, _, _ in pts])
    sp = np.diff(centers)
    if sp.mean() <= 0 or sp.std()/sp.mean() > cv_max:   # 등간격이 아니면 skip
        return res, False
    idx = np.arange(len(centers))

    def local_pitch(j):
        lo = max(0, j-3); hi = min(len(centers), j+4)
        return float(np.polyfit(np.arange(lo, hi), centers[lo:hi], 1)[0])
    pitch = np.array([local_pitch(j) for j in idx])
    M = pitch / pitch.mean()                            # 국소 배율 (오른쪽>1)
    out = list(res)
    for (c, p, ri), m in zip(pts, M):
        s, e, _ = res[ri]
        out[ri] = (s, e, float(p / m))
    return out, True


def average_results(res_list):
    """여러 이미지의 측정결과(같은 ROI)를 선별로 평균 → 랜덤오차 감소 (논문 10장평균)."""
    base = res_list[0]
    out = []
    for i, (s, e, _) in enumerate(base):
        vals = [r[i][2] for r in res_list if i < len(r) and r[i][2] is not None]
        out.append((s, e, float(np.mean(vals)) if vals else None))
    return out


# ══════════════════════════════════════════════════════════════
#  개선부: masked-background 측정 + measure_v2
# ══════════════════════════════════════════════════════════════
def _masked_background(prof, segs, margin, smooth=25):
    """검출된 선 구간(±margin px)을 마스킹하고 배경 픽셀만 선형 보간한 배경."""
    n = len(prof)
    mask = np.zeros(n, bool)
    for (s, e, c, w) in segs:
        mask[max(0, int(s) - margin):min(n, int(e) + margin + 1)] = True
    if mask.all() or (~mask).sum() < 8:
        return np.zeros_like(prof, dtype=np.float32)
    x = np.arange(n, dtype=float)
    bg = np.interp(x, x[~mask], np.asarray(prof, float)[~mask])
    return uniform_filter1d(bg.astype(np.float32), size=smooth, mode="reflect")


def drop_ghosts_by_pitch(res, n_expected=22):
    """등간격 게이지 평가 전용: 초과 검출을 피치 CV 최소화로 제거.
    (실제 크랙에는 사용하지 말 것 — 등간격 가정이 성립하지 않음)"""
    res = list(res)
    while len(res) > n_expected:
        cs = np.array([(s + e) / 2 for (s, e, m) in res])
        best = (np.inf, None)
        for i in range(len(cs)):
            sp = np.diff(np.sort(np.delete(cs, i)))
            cv = sp.std() / sp.mean()
            if cv < best[0]:
                best = (cv, i)
        res.pop(best[1])
    return res


def measure_v2(img, BC, BCY, CR, CRY, nrows=10, masked_bg=True, esf_sigma=True,
               band_frac=0.85, win_frac=0.6):
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

    def fwcal(f):
        if f is None:
            return None
        if f < pxthr:
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
        margin = int(round(1.5 * FWpsf))
        bg = _masked_background(raw_band, segs, margin)
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
        res.append((s, e, mm))

    cx, cy, cw, ch = CR
    info = dict(bc=bc, cr=g[cy:cy + ch, cx:cx + cw], BCY=BCY, CRY=CRY,
                bars=bars, grades=grades, bcl=bcl, crl2=crl2, segs=segs,
                masked_bg=masked_bg, sigma=sig, FWpsf=FWpsf, pxmm=pxmm)
    return pxmm, sig, res, info


def measure_crack(img, BC, BCY, CR, CRY, masked_bg=True, pick="prominent",
                  band_frac=0.85, win_frac=0.6):
    """
    실사용 API: 균열 '하나'의 폭 '하나'(mm)를 반환.

    내부적으로 measure_v2와 동일한 파이프라인을 돌린 뒤, 검출된 세그먼트 중
    하나를 선택한다.
      pick="prominent" : 프로파일 적분 면적(∑ 진폭)이 가장 큰 선 (기본).
                          피크 진폭이 아니라 면적 기준 — 텍스처 유령 세그먼트는
                          면적이 작아 자연 배제되고, 진하고 넓은 균열이 선택됨.
      pick="center"    : ROI 가로 중심에 가장 가까운 선 (촬영 UI가 균열을
                          중앙 정렬시키는 경우)
      pick=int         : 세그먼트 인덱스 직접 지정

    반환: (width_mm | None, detail)
      detail = dict(seg=(s,e), n_segments, pxmm, sigma, all=res, masked_bg)
    주의: 폭 값 자체는 여전히 밴드 전 행의 행별 FWHM median(해상 구간) 또는
    행-median 프로파일 단일 피팅(서브해상)으로 계산된다 — '값 하나'는 선택의
    문제이지 집계를 생략한다는 뜻이 아니다.
    """
    pxmm, sig, res, info = measure_v2(img, BC, BCY, CR, CRY, masked_bg=masked_bg,
                                      band_frac=band_frac, win_frac=win_frac)
    segs = info["segs"]
    if not segs:
        return None, dict(seg=None, n_segments=0, pxmm=pxmm, sigma=sig,
                          all=res, masked_bg=masked_bg)
    crl2c = np.clip(np.asarray(info["crl2"]), 0, None)
    if isinstance(pick, int):
        i = max(0, min(len(segs) - 1, pick))
    elif pick == "center":
        cx0 = len(crl2c) / 2.0
        i = int(np.argmin([abs((s + e) / 2 - cx0) for (s, e, c, w) in segs]))
    else:  # "prominent" — 적분 면적 최대 세그먼트
        i = int(np.argmax([float(crl2c[int(s):int(e) + 1].sum()) for (s, e, c, w) in segs]))
    s, e, mm = res[i]
    return mm, dict(seg=(s, e), n_segments=len(segs), pxmm=pxmm, sigma=sig,
                    all=res, masked_bg=masked_bg)


TRUTH = [0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0,1.5,2.0,2.5]


GUI_MAX = 1000   # GUI 표시 최대 변(px). 큰 사진은 이 크기로 축소해 보여줌.


def _disp_scale(h, w):
    """화면에 맞게 축소할 배율(≤1). 긴 변이 GUI_MAX 이하가 되도록."""
    return min(1.0, GUI_MAX / float(max(h, w)))


def select_roi_on_image(gray, window_title="ROI"):
    """드래그로 ROI 선택 → 원본좌표 (x,y,w,h). 큰 사진은 축소표시 후 좌표 환산."""
    img = gray if gray.ndim == 3 else cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    h, w = img.shape[:2]
    sc = _disp_scale(h, w)
    disp = cv2.resize(img, (int(w*sc), int(h*sc)), interpolation=cv2.INTER_AREA) if sc < 1 else img
    cv2.namedWindow(window_title, cv2.WINDOW_AUTOSIZE)
    r = cv2.selectROI(window_title, disp, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(window_title)
    return tuple(int(round(v / sc)) for v in r)   # 원본 해상도로 환산


def pick_scan_y(crop_gray, window_title="scan Y"):
    """크롭 위에서 스캔라인 Y 클릭(미클릭 시 세로 중앙) → 원본좌표 y(int)."""
    h, w = crop_gray.shape[:2]
    sc = _disp_scale(h, w)
    base = cv2.cvtColor(crop_gray, cv2.COLOR_GRAY2BGR)
    disp0 = cv2.resize(base, (int(w*sc), int(h*sc)), interpolation=cv2.INTER_AREA) if sc < 1 else base
    state = {'y': disp0.shape[0] // 2}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            state['y'] = int(y)
    try:
        cv2.namedWindow(window_title, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(window_title, on_mouse)
        while True:
            vis = disp0.copy()
            cv2.line(vis, (0, state['y']), (vis.shape[1], state['y']), (0, 0, 255), 1)
            cv2.imshow(window_title, vis)
            key = cv2.waitKey(20) & 0xFF
            if key in (13, 32, 27):   # Enter/Space/Esc
                break
        cv2.destroyWindow(window_title)
    except Exception:
        pass
    return int(round(state['y'] / sc))   # 원본 해상도로 환산


def export_results(export_prefix, tag, img, res, t_use, errs, pxmm, sig, n, n_truth):
    """측정 결과를 CSV(누적) + XLSX로 내보낸다. multi/단일 모드 공용."""
    import csv, os, datetime
    rows = []
    for i,(s,e,p) in enumerate(res):
        t  = t_use[i] if i < len(t_use) else None
        es = round(p, 4) if p else None
        er = round(p - t, 4) if (p and t) else None
        rows.append({"tag": tag, "truth_mm": t, "est_mm": es, "err_mm": er})
    mae  = float(np.mean(errs)) if errs else None
    mx   = float(max(errs))     if errs else None
    rmse = float(np.sqrt(np.mean([x*x for x in errs]))) if errs else None
    mape = float(np.mean([abs(r["err_mm"])/r["truth_mm"]*100
                          for r in rows if r["err_mm"] is not None and r["truth_mm"]])) if errs else None
    summ = {"tag": tag, "n_detected": n, "n_truth": n_truth,
            "px_per_mm": round(pxmm, 4), "sigma_px": round(sig, 4),
            "MAE_mm": round(mae,4) if mae is not None else None,
            "max_mm": round(mx,4)  if mx  is not None else None,
            "RMSE_mm": round(rmse,4) if rmse is not None else None,
            "MAPE_pct": round(mape,2) if mape is not None else None,
            "image": img,
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds")}

    pw_path = f"{export_prefix}_perwidth.csv"
    write_header = not os.path.exists(pw_path)
    with open(pw_path, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["tag","truth_mm","est_mm","err_mm"])
        if write_header: w.writeheader()
        w.writerows(rows)
    sm_path = f"{export_prefix}_summary.csv"
    write_header = not os.path.exists(sm_path)
    with open(sm_path, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(summ.keys()))
        if write_header: w.writeheader()
        w.writerow(summ)
    print(f"[내보내기] {pw_path} (누적)  ·  {sm_path} (누적)")

    try:
        import pandas as pd
        xlsx_path = f"{export_prefix}.xlsx"
        dfp = pd.read_csv(pw_path)
        dfs = pd.read_csv(sm_path)
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as xw:
            dfs.to_excel(xw, sheet_name="summary", index=False)
            dfp.to_excel(xw, sheet_name="per_width", index=False)
            try:
                piv = dfp.pivot_table(index="truth_mm", columns="tag",
                                      values="est_mm", aggfunc="mean")
                piv.to_excel(xw, sheet_name="pivot_est")
            except Exception:
                pass
        print(f"[내보내기] {xlsx_path} (summary/per_width/pivot_est 시트)")
    except Exception as ex:
        print(f"  (XLSX 생략: {ex})")


def report_and_export(img, res, pxmm, sig, TRUTH, export_prefix=None, tag=None):
    """결과 콘솔 출력 + (export 지정 시) 파일 내보내기. 측정값 리스트를 받아 처리."""
    import os
    n = len(res)
    print(f"\n검출 {n}개  (정답 {len(TRUTH)}개)")
    off = len(TRUTH) - n
    t_use = TRUTH[off:] if off > 0 else TRUTH[:n]
    errs = []
    print(f"{'#':>2} {'px':>10} {'정답':>5} {'측정':>6} {'오차':>7}")
    for i,(s,e,p) in enumerate(res):
        t = t_use[i] if i < len(t_use) else None
        er = (p - t) if (p and t) else None
        if er is not None: errs.append(abs(er))
        print(f"{i+1:>2} {s:>4}~{e:<4} {t if t else 0:>5.2f} {p if p else 0:>6.2f} {er if er is not None else 0:>+7.3f}")
    if errs:
        miss = f"가는바 {off}개 누락" if off>0 else "전부 검출"
        print(f"\nMAE={np.mean(errs):.4f}mm  최대={max(errs):.4f}mm  (검출 {n}/{len(TRUTH)}, {miss})")
    if export_prefix:
        t = tag if tag else os.path.splitext(os.path.basename(img))[0]
        export_results(export_prefix, t, img, res, t_use, errs, pxmm, sig, n, len(TRUTH))


def export_profiles(export_prefix, tag, img, CR, CRY, info):
    """crack ROI의 위치별 프로파일을 CSV로 저장 (배경제거 전/후, center row/전체행 median).
    저장 컬럼(각 tag별):
      <tag>_center_raw   : center row의 255 - I            (배경제거 전)
      <tag>_median_raw   : 전체 밴드 행 median의 255 - I    (배경제거 전)
      <tag>_median_crl2  : measure_v2의 crl2 (배경제거 후, 측정에 실제 쓰인 프로파일)
    파일: <export_prefix>_profiles_<tag>.csv  (이미지별 개별 파일)
    """
    import csv, os
    g = load_gray(img)
    if g is None:
        print(f"  (프로파일 생략: {img} 로드 실패)"); return
    cx, cy, cw, ch = CR
    roi = g[cy:cy+ch, cx:cx+cw].astype(float)
    # center row (CRY는 ROI 내 상대 y)
    yy = int(np.clip(CRY, 0, roi.shape[0]-1))
    center_raw = 255.0 - roi[yy, :]
    # 전체 밴드 median (measure_v2와 같은 band_frac=0.85 밴드)
    b0, b1 = _band(ch, CRY, 0.85)
    median_raw = 255.0 - np.median(roi[b0:b1+1, :], axis=0)
    # 배경제거 후 (측정에 실제 쓰인 프로파일)
    crl2 = np.asarray(info.get('crl2'))
    W = len(center_raw)
    # 길이 정합 (crl2가 다르면 자름/패딩)
    if len(crl2) != W:
        crl2 = np.interp(np.linspace(0,1,W), np.linspace(0,1,len(crl2)), crl2)

    path = f"{export_prefix}_profiles_{tag}.csv"
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["position",
                    f"{tag}_center_raw",
                    f"{tag}_median_raw",
                    f"{tag}_median_crl2"])
        for x in range(W):
            w.writerow([x,
                        round(float(center_raw[x]), 3),
                        round(float(median_raw[x]), 3),
                        round(float(crl2[x]), 3)])
    print(f"[프로파일] {path} ({W}px)")


def gui_pick_rois(img):
    """한 이미지에서 바코드 ROI/Y, 크랙 ROI/Y를 GUI로 지정 → (BC,BCY,CR,CRY)."""
    gray = load_gray(img)
    if gray is None:
        print("이미지 로드 실패:", img); return None
    print(f"\n=== {img} ===")
    print("[1] 바코드 영역 드래그 → Enter/Space")
    BC = select_roi_on_image(gray, f"{img}  1) Barcode ROI")
    bc_crop = gray[BC[1]:BC[1]+BC[3], BC[0]:BC[0]+BC[2]]
    print("[2] 바코드 스캔라인 Y 클릭")
    BCY = pick_scan_y(bc_crop, f"{img}  2) Barcode scan Y")
    print("[3] 크랙(자) 영역 드래그 → Enter/Space")
    CR = select_roi_on_image(gray, f"{img}  3) Crack ROI")
    cr_crop = gray[CR[1]:CR[1]+CR[3], CR[0]:CR[0]+CR[2]]
    print("[4] 크랙 스캔라인 Y 클릭")
    CRY = pick_scan_y(cr_crop, f"{img}  4) Crack scan Y")
    return BC, BCY, CR, CRY


def draw(img_path, BC, BCY, CR, CRY, pxmm, sig, res, info, save):
    import matplotlib; matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    GC = {'T1':'#00FFFF','T2':'#00FF00','T3':'#FFFF00','T4':'#FF8800','T5':'#FF3333'}
    fig = plt.figure(figsize=(19, 10)); fig.patch.set_facecolor('#1a1a1a')
    fig.suptitle(f"VisionCrack (fixed)   px/mm={pxmm:.2f}   sigma={sig:.2f}px   detected {len(res)}/{len(TRUTH)}",
                 color='white', fontsize=14, fontweight='bold')
    gs = fig.add_gridspec(3, 10, height_ratios=[1.3, 1.0, 1.5], hspace=0.42, wspace=0.45)

    # ── row0: 바코드 이미지 | 크랙 이미지 ──
    axb = fig.add_subplot(gs[0, :5]); axb.imshow(info['bc'], cmap='gray', aspect='auto')
    axb.axhline(BCY, color='r', lw=1.0, ls='--')
    for (s,e,c,w),gr in zip(info['bars'], info['grades']):
        axb.axvline(c, color=GC.get(gr,'w'), lw=0.7, alpha=0.7)
        axb.text(c, BCY, gr, color='k', fontsize=6, ha='center', va='center', fontweight='bold',
                 bbox=dict(facecolor=GC.get(gr,'w'), edgecolor='none', boxstyle='round,pad=0.1'))
    axb.set_title(f"Barcode  [{len(info['bars'])} bars]", color='white', fontsize=10); axb.tick_params(colors='gray')

    axc = fig.add_subplot(gs[0, 5:]); axc.imshow(info['cr'], cmap='gray', aspect='auto')
    axc.axhline(CRY, color='r', lw=1.0, ls='--')
    for (s,e,p) in res:
        axc.axvline((s+e)//2, color='#33ff33', lw=0.7, ls=':', alpha=0.8)
    axc.set_title("Crack Region", color='white', fontsize=10); axc.tick_params(colors='gray')

    # ── row1: 바코드 풀 프로파일 (등급색 밴드) ──
    axbp = fig.add_subplot(gs[1, :]); axbp.set_facecolor('#111111')
    bcl = info['bcl']; axbp.plot(np.arange(len(bcl)), bcl, color='white', lw=1.0)
    for (s,e,c,w),gr in zip(info['bars'], info['grades']):
        axbp.axvspan(s, e, alpha=0.4, color=GC.get(gr,'w'))
        axbp.text(c, bcl[s:e+1].max()+6, gr, color=GC.get(gr,'w'), fontsize=7, ha='center', va='bottom', fontweight='bold')
    axbp.set_xlim(0, len(bcl)-1); axbp.set_title("Barcode Full Profile", color='white', fontsize=10)
    axbp.set_ylabel("Intensity (inv)", color='gray'); axbp.tick_params(colors='gray'); axbp.grid(True, alpha=0.2)

    # ── row2: 크랙 프로파일 + 결과표 ──
    axcp = fig.add_subplot(gs[2, :6]); axcp.set_facecolor('#111111')
    crl = info['crl2']; axcp.plot(np.arange(len(crl)), crl, color='tomato', lw=1.0)
    axcp.fill_between(np.arange(len(crl)), 0, crl, alpha=0.12, color='tomato')
    off = len(TRUTH) - len(res); t_use = TRUTH[off:] if off>0 else TRUTH[:len(res)]
    for i,(s,e,p) in enumerate(res):
        axcp.axvspan(s, e, alpha=0.25, color='yellow')
        pk = float(crl[s:e+1].max())
        axcp.annotate(f"#{i+1}", xy=((s+e)//2, pk), xytext=((s+e)//2, pk+25),
                      color='yellow', fontsize=7, ha='center', fontweight='bold')
        axcp.text((s+e)//2, pk+50, f"{p:.2f}" if p else "?", color='#66ff66', fontsize=7,
                  ha='center', fontweight='bold')
    axcp.set_xlim(0, len(crl)-1); axcp.set_ylim(0, max(crl.max()*1.6, 120))
    axcp.set_title("Crack Profile  (measured, mm)", color='white', fontsize=10, pad=24)
    axcp.set_xlabel("Pixel X", color='gray'); axcp.tick_params(colors='gray'); axcp.grid(True, alpha=0.2)

    axt = fig.add_subplot(gs[2, 6:]); axt.axis('off'); axt.set_xlim(0,1); axt.set_ylim(0,1)
    errs = []; rows = [" #  true   meas    err"]
    for i,(s,e,p) in enumerate(res):
        t = t_use[i] if i < len(t_use) else None
        er = (p-t) if (p and t) else None
        if er is not None: errs.append(abs(er))
        rows.append(f"{i+1:>2}   {t if t else 0:>4.2f}   {p if p else 0:>5.2f}  {er if er is not None else 0:>+6.3f}")
    mae = np.mean(errs) if errs else 0
    axt.text(0.0, 1.0, "Measurement Results", color='white', fontsize=12, fontweight='bold', va='top', family='monospace')
    half = (len(rows)+1)//2
    axt.text(0.0, 0.90, "\n".join(rows[:half]), color='#9fff9f', fontsize=9.5, family='monospace', va='top')
    axt.text(0.5, 0.90, "\n".join([rows[0]]+rows[half:]), color='#9fff9f', fontsize=9.5, family='monospace', va='top')
    axt.text(0.0, 0.16, f"MAE = {mae:.4f} mm    max = {max(errs) if errs else 0:.4f} mm    detected {len(res)}/{len(TRUTH)}",
             color='#ffd24d', fontsize=11, fontweight='bold', va='top', family='monospace')
    plt.savefig(save, dpi=100, facecolor='#1a1a1a', bbox_inches='tight'); plt.close()
    print(f"[그림] {save}")


if __name__ == "__main__":
    import argparse
    def _roi(s): return tuple(int(v) for v in s.split(","))
    ap = argparse.ArgumentParser()
    ap.add_argument("image", nargs="?", default="image_black_white/Left_1.png")
    ap.add_argument("--bc-roi", type=_roi, default=None)
    ap.add_argument("--bc-y", type=int, default=None)
    ap.add_argument("--cr-roi", type=_roi, default=None)
    ap.add_argument("--cr-y", type=int, default=None)
    ap.add_argument("--save", default="vc_result.png")
    ap.add_argument("--dpi", type=int, default=2400, help="PDF 렌더 DPI")
    ap.add_argument("--truth", default=None, help="쉼표구분 정답 mm (없으면 기본 14개: 0.5~2.5)")
    ap.add_argument("--keystone", action="store_true",
                    help="등간격 게이지 원근(키스톤) 보정: 국소 피치로 위치별 배율 제거")
    ap.add_argument("--no-mask", action="store_true",
                    help="masked-bg 개선 끄기 (v1과 동일 동작)")
    ap.add_argument("--crack", action="store_true",
                    help="단일 균열 모드: 가장 뚜렷한 선 하나의 폭만 출력")
    ap.add_argument("--avg", default=None,
                    help="다중이미지 평균: 같은 ROI의 추가 이미지들(쉼표구분). 논문 10장평균 재현")
    ap.add_argument("--band-frac", type=float, default=0.85,
                    help="측정/검출 스캔밴드 높이비 (rev4 기본 0.85; rev3 동작=0.35)")
    ap.add_argument("--win-frac", type=float, default=0.6,
                    help="FWHM 측정창 = 세그폭의 ±win-frac (rev4 기본 0.6; rev3 동작=1.0)")
    ap.add_argument("--tag", default=None,
                    help="조건 이름 (예: T_wood, opaque_A4). CSV·엑셀 내보낼 때 행 식별용")
    ap.add_argument("--export", default=None,
                    help="결과 내보내기 파일 접두사 (예: results). "
                         "results_perwidth.csv(누적) / results_summary.csv(누적) / results.xlsx 생성")
    ap.add_argument("--multi", default=None,
                    help="다중 개별 측정: 여러 이미지(쉼표구분)를 순서대로 GUI로 ROI를 "
                         "각각 지정하고, 각 이미지 결과를 개별 행으로 --export에 누적. "
                         "예: --multi 5/W1.png,5/W2.png,...,5/W10.png")
    ap.add_argument("--multi-dir", default=None, dest="multi_dir",
                    help="폴더 안 이미지를 자동 스캔해 --multi처럼 각각 측정. "
                         "이름 상관없이 정렬 순서대로. 예: --multi-dir 5")
    a = ap.parse_args()
    PDF_DPI = a.dpi
    if a.truth:
        TRUTH = [float(x) for x in a.truth.split(",")]
    img = a.image
    BC, BCY, CR, CRY = a.bc_roi, a.bc_y, a.cr_roi, a.cr_y

    # ── 다중 개별 측정 모드: 각 이미지마다 ROI를 새로 지정하고 개별 저장 ──
    if a.multi or a.multi_dir:
        if a.multi_dir:
            import glob as _glob, os as _os
            exts = ("*.png","*.jpg","*.jpeg","*.bmp","*.tif","*.tiff",
                    "*.PNG","*.JPG","*.JPEG","*.BMP","*.TIF","*.TIFF")
            found = []
            for ext in exts:
                found += _glob.glob(_os.path.join(a.multi_dir, ext))
            mimgs = sorted(set(found))
            if not mimgs:
                print(f"이미지 없음: {a.multi_dir} 폴더에 png/jpg 등이 없습니다."); sys.exit(1)
            print(f"[폴더 스캔] {a.multi_dir} → {len(mimgs)}장 발견")
            for p in mimgs: print(f"   - {p}")
        else:
            mimgs = [s.strip() for s in a.multi.split(",") if s.strip()]
        if not a.export:
            print("경고: --multi는 보통 --export와 함께 씁니다 (결과 누적 저장). "
                  "그림만 저장하고 진행합니다.")
        print(f"[다중 모드] {len(mimgs)}장 개별 측정 시작")
        ok = 0
        for idx, mimg in enumerate(mimgs, 1):
            print(f"\n────────── ({idx}/{len(mimgs)})  {mimg} ──────────")
            picked = gui_pick_rois(mimg)
            if picked is None:
                print(f"  건너뜀: {mimg} (로드 실패)")
                continue
            bc, bcy, cr, cry = picked
            try:
                pxmm_i, sig_i, res_i, info_i = measure_v2(
                    mimg, bc, bcy, cr, cry, masked_bg=not a.no_mask,
                    band_frac=a.band_frac, win_frac=a.win_frac)
            except Exception as ex:
                print(f"  측정 실패: {mimg} — {ex}")
                continue
            if a.keystone:
                res_i, applied = keystone_correct(res_i)
                print("키스톤 보정:", "적용됨" if applied else "미적용(등간격 아님)")
            if len(res_i) > len(TRUTH):
                nd = len(res_i) - len(TRUTH)
                res_i = drop_ghosts_by_pitch(res_i, len(TRUTH))
                print(f"유령 검출 제거: {nd}개")
            # 이미지별 결과 그림 저장 (파일명에 인덱스)
            save_i = a.save.replace(".png", f"_{idx:02d}.png") if a.save.endswith(".png") \
                     else f"{a.save}_{idx:02d}.png"
            draw(mimg, bc, bcy, cr, cry, pxmm_i, sig_i, res_i, info_i, save_i)
            # 이미지별 tag = --tag 있으면 tag_인덱스, 없으면 파일명
            import os as _os
            base = _os.path.splitext(_os.path.basename(mimg))[0]
            tag_i = f"{a.tag}_{base}" if a.tag else base
            report_and_export(mimg, res_i, pxmm_i, sig_i, TRUTH,
                              export_prefix=a.export, tag=tag_i)
            if a.export:
                export_profiles(a.export, tag_i, mimg, cr, cry, info_i)
            ok += 1
        print(f"\n[다중 모드 완료] {ok}/{len(mimgs)}장 처리·저장")
        sys.exit(0)

    # ── 좌표 없으면 GUI로 직접 선택 ──
    if BC is None or CR is None:
        gray = load_gray(img)
        if gray is None:
            print("이미지 로드 실패:", img); sys.exit(1)
        print("\n[1] 바코드 영역 드래그 → Enter/Space")
        BC = select_roi_on_image(gray, "1) Barcode ROI - 드래그후 Enter")
        bc_crop = gray[BC[1]:BC[1]+BC[3], BC[0]:BC[0]+BC[2]]
        print("[2] 바코드 스캔라인 Y 클릭")
        BCY = pick_scan_y(bc_crop, "2) Barcode scan Y")
        print("[3] 크랙(자) 영역 드래그 → Enter/Space")
        CR = select_roi_on_image(gray, "3) Crack ROI - 드래그후 Enter")
        cr_crop = gray[CR[1]:CR[1]+CR[3], CR[0]:CR[0]+CR[2]]
        print("[4] 크랙 스캔라인 Y 클릭")
        CRY = pick_scan_y(cr_crop, "4) Crack scan Y")
        print("="*60)
        print("  [좌표 저장] 다음엔 GUI 없이:")
        print(f"    --bc-roi {BC[0]},{BC[1]},{BC[2]},{BC[3]}  --bc-y {BCY}")
        print(f"    --cr-roi {CR[0]},{CR[1]},{CR[2]},{CR[3]}  --cr-y {CRY}")
        print("="*60)

    _measure = lambda *args: measure_v2(*args, masked_bg=not a.no_mask,
                                        band_frac=a.band_frac, win_frac=a.win_frac)
    pxmm, sig, res, info = _measure(img, BC, BCY, CR, CRY)
    if a.crack:
        wmm, dd = measure_crack(img, BC, BCY, CR, CRY, masked_bg=not a.no_mask,
                                band_frac=a.band_frac, win_frac=a.win_frac)
        if wmm is not None:
            print(f"\n[단일 균열] width = {wmm:.3f} mm   seg={dd['seg']}   "
                  f"(검출 {dd['n_segments']}개 중 선택, sigma={sig:.2f}px)")
        else:
            print("\n[단일 균열] 측정 실패 — 선 미검출")
        draw(img, BC, BCY, CR, CRY, pxmm, sig, res, info, a.save)
        sys.exit(0)
    if a.avg:                                          # 다중이미지 평균 (논문 10장평균)
        imgs = [s.strip() for s in a.avg.split(",") if s.strip()]
        res_list = [res]
        for im in imgs:
            try:
                _, _, r2, _ = _measure(im, BC, BCY, CR, CRY)
                res_list.append(r2)
            except Exception as ex:
                print(f"  (평균 제외: {im} — {ex})")
        res = average_results(res_list)
        print(f"다중이미지 평균: {len(res_list)}장 ({1+len(imgs)} 요청)")
    if a.keystone:
        res, applied = keystone_correct(res)
        print("키스톤 보정:", "적용됨" if applied else "미적용(등간격 아님)")
    if len(res) > len(TRUTH):
        n_drop = len(res) - len(TRUTH)
        res = drop_ghosts_by_pitch(res, len(TRUTH))
        print(f"유령 검출 제거: {n_drop}개 (피치 일관성 기준 — 등간격 게이지 평가 전용)")
    draw(img, BC, BCY, CR, CRY, pxmm, sig, res, info, a.save)
    report_and_export(img, res, pxmm, sig, TRUTH,
                      export_prefix=a.export, tag=a.tag)
    if a.export:
        import os as _os2
        _tag = a.tag if a.tag else _os2.path.splitext(_os2.path.basename(img))[0]
        export_profiles(a.export, _tag, img, CR, CRY, info)
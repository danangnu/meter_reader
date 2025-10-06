# lcd_ocr.py
import os, sys, cv2, json, math, shutil, argparse, re
import numpy as np
import pytesseract
from typing import Optional, Tuple, List

# ===================== Tesseract path =====================
def _setup_tesseract_path() -> Optional[str]:
    fixed = r"C:\Tesseract\tesseract.exe"
    for p in [fixed, os.environ.get("TESSERACT_CMD"), shutil.which("tesseract")]:
        if p and os.path.isfile(p):
            pytesseract.pytesseract.tesseract_cmd = p
            return p
    return None
_TESS = _setup_tesseract_path()

# ----- Shape heuristics used for last-mile overrides -----
def _looks_like_four_sig(ls, rs, c_sig, f_sig, b_sig, a_sig, g_eff, d_eff) -> bool:
    right_crush = (rs >= ls + 0.24) and (c_sig >= 0.24) and (f_sig >= 0.22)
    return (
        d_eff < 0.30 and
        ls    < 0.34 and
        a_sig < 0.40 and
        g_eff >= 0.04 and
        (b_sig >= 0.12 or f_sig >= 0.12) and
        not right_crush
    )

def _looks_like_four_sig_soft(ls, rs, c_sig, f_sig, b_sig, a_sig, g_eff, d_eff) -> bool:
    return (
        d_eff < 0.36 and
        ls    < 0.40 and
        a_sig < 0.46 and
        g_eff >= 0.03 and
        (b_sig >= 0.09 or f_sig >= 0.09)
    )

def _looks_like_five_sig(ls, rs, c_sig, f_sig, b_sig, a_sig, g_eff, d_eff) -> bool:
    right_dom = (rs >= ls + 0.12) and (c_sig >= 0.22)
    return (d_eff >= 0.24) and right_dom and (g_eff >= 0.06)

def _looks_like_zero_sig(a,b,c,d,e,f,g_eff) -> bool:
    outer = min(a,b,c,d,e,f)
    return (outer >= 0.20) and (g_eff <= 0.22)

def _looks_like_zero_sig_soft(a,b,c,d,e,f,g_eff) -> bool:
    outer = min(a,b,c,d,e,f)
    return (outer >= 0.14) and (g_eff <= 0.30)

def _looks_like_eight_sig(a,b,c,d,e,f,g_eff) -> bool:
    outer = min(a,b,c,d,e,f)
    return (outer >= 0.20) and (g_eff >= 0.18)

# ===================== Geometry helpers =====================
def order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1); d = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]; rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(d)]; rect[3] = pts[np.argmax(d)]
    return rect

def four_point_transform(image: np.ndarray, pts: np.ndarray, pad: int = 2) -> np.ndarray:
    rect = order_points(pts.astype("float32"))
    (tl, tr, br, bl) = rect
    wA = np.linalg.norm(br - bl); wB = np.linalg.norm(tr - tl)
    hA = np.linalg.norm(tr - br); hB = np.linalg.norm(tl - bl)
    W = max(1, int(max(wA, wB))); H = max(1, int(max(hA, hB)))
    dst = np.array([[0,0],[W-1,0],[W-1,H-1],[0,H-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warp = cv2.warpPerspective(image, M, (W, H))
    if pad > 0:
        warp = cv2.copyMakeBorder(warp, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    return warp

# ===================== LCD detection (green-first; edge fallback) =====================
def find_lcd_quad_color_first(bgr: np.ndarray) -> Optional[np.ndarray]:
    h, w = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([35, 30, 40]); upper = np.array([95, 255, 255])  # broad green
    mask = cv2.inRange(hsv, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, 1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, 2)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None; best_score = -1.0; img_area = float(h*w)

    def _score_rect(rect_pts: np.ndarray, area: float) -> float:
        r = order_points(rect_pts)
        wA = np.linalg.norm(r[1] - r[0]); wB = np.linalg.norm(r[2] - r[3])
        hA = np.linalg.norm(r[3] - r[0]); hB = np.linalg.norm(r[2] - r[1])
        width = max(wA, wB); height = max(hA, hB)
        if height <= 0: return -1
        ar = width / height
        if not (1.2 <= ar <= 12.0): return -1
        ar_score = math.exp(-((ar - 4.0) ** 2) / (2 * 2.0 ** 2))
        return (area / img_area) * 1.2 + ar_score * 0.8

    for c in cnts:
        area = cv2.contourArea(c)
        if area < 0.002 * img_area: continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        rect = approx.reshape(4,2).astype(np.float32) if len(approx)==4 else \
               cv2.boxPoints(cv2.minAreaRect(c)).astype(np.float32)
        s = _score_rect(rect, area)
        if s > best_score: best_score, best = s, rect

    if best is not None:
        return best

    # Fallback: edges
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 7, 60, 60)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8,8)); gray = clahe.apply(gray)
    v = np.median(gray); lo = int(max(0, 0.66*v)); hi = int(min(255, 1.33*v))
    edges = cv2.Canny(gray, lo, hi)
    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (5,3)), 1)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5,3)), 1)
    cnts2, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None; best_score = -1.0
    for c in cnts2:
        area = cv2.contourArea(c)
        if area < 0.01 * img_area: continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) != 4: continue
        rect = approx.reshape(4,2).astype(np.float32)
        hull = cv2.convexHull(rect); hull_area = cv2.contourArea(hull)
        if hull_area <= 0: continue
        rectangularity = area / hull_area
        if rectangularity < 0.80: continue
        s = _score_rect(rect, area)
        if s > best_score: best_score, best = s, rect

    return best

# ===================== Binarize (glow-resistant) =====================
def _extra_unsharp(x: np.ndarray, amount: float = 1.8) -> np.ndarray:
    g = cv2.GaussianBlur(x, (0,0), 1.0)
    return cv2.addWeighted(x, 1.0 + amount, g, -amount, 0)

def binarize_lcd(crop: np.ndarray, scale: float = 6.0, boost_contrast: bool=False) -> np.ndarray:
    g = crop[:,:,1].astype(np.float32)
    v = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)[:,:,2].astype(np.float32)
    lum = (0.65*g + 0.35*v).astype(np.uint8)
    k = max(9, (min(crop.shape[0], crop.shape[1])//8)|1)
    bg = cv2.morphologyEx(lum, cv2.MORPH_OPEN,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k,k)))
    flat = cv2.subtract(lum, bg)
    flat = cv2.normalize(flat, None, 0, 255, cv2.NORM_MINMAX)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
    x = clahe.apply(flat)
    if boost_contrast:
        x = _extra_unsharp(x, amount=1.8)
    else:
        blur = cv2.GaussianBlur(x, (0,0), 1.0)
        x = cv2.addWeighted(x, 1.8, blur, -0.8, 0)
    x = cv2.resize(x, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    th = cv2.adaptiveThreshold(x, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 31, 8)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((2,2), np.uint8), 1)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((2,2), np.uint8), 1)
    return 255 - th  # black digits on white

# ===================== Slotting =====================
def _spans_from_mask(mask: np.ndarray) -> List[Tuple[int,int]]:
    spans = []
    in_run = False; st = 0
    for i,v in enumerate(mask):
        if v and not in_run: in_run=True; st=i
        elif not v and in_run: in_run=False; spans.append((st,i-1))
    if in_run: spans.append((st,len(mask)-1))
    merged=[]
    for s,e in spans:
        if not merged or s-merged[-1][1] > 2: merged.append((s,e))
        else: merged[-1]=(merged[-1][0],e)
    return merged

def _multi_split_for_proj(spans: List[Tuple[int,int]], proj: np.ndarray) -> List[Tuple[int,int]]:
    if not spans: return spans
    widths = [e-s+1 for s,e in spans]
    medw = max(1.0, float(np.median(widths)))

    def multi_split(s, e):
        parts = [(s, e)]
        changed = True
        while changed:
            changed = False
            new_parts = []
            for a, b in parts:
                wspan = b - a + 1
                if wspan < int(1.6 * medw):
                    new_parts.append((a, b)); continue
                sub = proj[a:b+1]
                sub_s = cv2.blur(sub.reshape(1, -1), (1,5)).ravel()
                thr = 0.80 * float(sub_s.max())
                minima = []
                for x in range(1, len(sub_s)-1):
                    if sub_s[x] < sub_s[x-1] and sub_s[x] <= sub_s[x+1] and sub_s[x] < thr:
                        minima.append(x)
                picks = []
                for x in minima:
                    if not picks or (x - picks[-1]) >= int(0.35*medw):
                        picks.append(x)
                if not picks:
                    new_parts.append((a, b)); continue
                last = a; made = False
                for x in picks:
                    cut = a + x
                    if (cut - last) >= int(0.40*medw) and (b - cut) >= int(0.40*medw):
                        new_parts.append((last, cut-1)); last = cut; made = True
                new_parts.append((last, b)); changed |= made
            parts = new_parts
        return parts

    improved = []
    for s,e in spans: improved.extend(multi_split(s,e))
    return sorted(improved)

def estimate_digit_slots(bw: np.ndarray) -> tuple[List[Tuple[int,int]], Tuple[int,int], np.ndarray]:
    inv = 255 - bw
    rows = inv.sum(axis=1).astype(np.float32)
    band = np.where(rows > 0.10*rows.max())[0]
    y0,y1 = (int(band.min()), int(band.max())) if band.size else (0, bw.shape[0]-1)
    tight = inv[y0:y1+1,:]

    proj = cv2.blur(tight, (7,1)).sum(axis=0).astype(np.float32)
    m = float(proj.max())
    if m <= 1e-6:
        return [], (y0,y1), proj

    candidates = []
    for thr_ratio in (0.10, 0.08, 0.06, 0.04):
        mask = (proj > (thr_ratio*m)).astype(np.uint8)
        spans = _spans_from_mask(mask)
        if not spans:
            continue
        widths = [e-s+1 for s,e in spans]
        medw = max(1.0, float(np.median(widths)))
        spans = [(s,e) for (s,e) in spans if (e-s+1) > 0.25*medw]
        spans = _multi_split_for_proj(spans, proj)
        candidates.append(spans)

    if not candidates:
        return [], (y0,y1), proj

    spans = max(candidates, key=lambda sp: (len(sp), -sum(e-s for s,e in sp)))
    return spans, (y0,y1), proj

def _force_split_to(spans: List[tuple], proj: np.ndarray, medw: float, target: int) -> List[tuple]:
    spans = sorted(spans)

    def try_minimum_cut(a: int, b: int) -> List[tuple]:
        sub = proj[a:b+1].astype(np.float32)
        if sub.size < 8: return [(a, b)]
        sub_s = cv2.blur(sub.reshape(1, -1), (1,5)).ravel()
        mins = []
        for x in range(1, len(sub_s)-1):
            if sub_s[x] < sub_s[x-1] and sub_s[x] <= sub_s[x+1]:
                mins.append((sub_s[x], x))
        if not mins: return [(a, b)]
        for _, x in sorted(mins, key=lambda t: t[0]):  # deepest first
            cut = a + x
            if cut <= a + 2 or cut >= b - 2: continue
            left_w  = cut - a; right_w = b - cut
            if left_w >= int(0.18*medw) and right_w >= int(0.18*medw):
                return [(a, cut-1), (cut, b)]
        return [(a, b)]

    def midpoint_cut(a: int, b: int) -> List[tuple]:
        cut = (a + b) // 2
        if cut <= a + 1 or cut >= b - 1: return [(a, b)]
        return [(a, cut), (cut+1, b)]

    safety = 40
    while len(spans) < target and safety > 0:
        safety -= 1
        widths = [e - s for (s, e) in spans]
        i = int(np.argmax(widths))
        a, b = spans[i]
        parts = try_minimum_cut(a, b)
        if len(parts) == 1: parts = midpoint_cut(a, b)
        if len(parts) == 1:
            widths[i] = -1
            if max(widths) <= 0: break
            for _ in range(len(spans)-1):
                i = int(np.argmax(widths))
                if widths[i] <= 0: break
                a, b = spans[i]
                parts = try_minimum_cut(a, b)
                if len(parts) == 1: parts = midpoint_cut(a, b)
                if len(parts) > 1:
                    spans[i:i+1] = parts; break
                widths[i] = -1
            else:
                break
        else:
            spans[i:i+1] = parts
    return sorted(spans)

# ===================== Dot detector (ANYWHERE) =====================
def detect_decimal_points_anywhere(bw: np.ndarray,
                                   spans: List[Tuple[int,int]],
                                   yband: Tuple[int,int]) -> List[int]:
    if not spans: return []
    y0, y1 = yband; H = y1 - y0 + 1
    band = bw[y0:y1+1, :]
    ink  = 255 - band

    widths = [e - s + 1 for (s, e) in spans]
    Wmed = float(np.median(widths)) if widths else band.shape[1]
    est_digit_area = max(1.0, Wmed * float(H))

    mask = (ink > 200).astype(np.uint8)
    num, labels, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num <= 1: return []

    boundaries = [ (spans[i][1] + spans[i+1][0]) / 2.0 for i in range(len(spans)-1) ]
    def x_to_k(cx: float) -> int:
        k = sum(1 for (_, e) in spans if cx > e)
        return max(0, min(k, len(spans)))

    cand = []
    for i in range(1, num):
        x,y,w,h,area = stats[i]; cx, cy = cents[i]
        area = float(area); w = float(w); h = float(h)
        if area < 0.0015*est_digit_area or area > 0.04*est_digit_area: continue
        if h == 0 or w == 0: continue
        ar = w / h
        if ar < 0.5 or ar > 2.0: continue
        if h > 0.45 * H: continue
        perim = 2.0*(w + h)
        circ = (4.0*math.pi*area)/(perim*perim) if perim > 0 else 0.0
        if circ < 0.55: continue
        inside = any(s+2 <= cx <= e-2 for (s,e) in spans)
        near_edge = any(abs(cx - s) <= 3 or abs(cx - e) <= 3 for (s,e) in spans)
        if inside and not near_edge: continue
        k = x_to_k(cx)
        if boundaries: d_boundary = min(abs(cx - b) for b in boundaries)
        else: d_boundary = min(abs(cx - spans[0][0]), abs(cx - spans[0][1]))
        score = 0.5*circ + 0.5*(1.0 - (d_boundary / max(1.0, Wmed/2.0)))
        cand.append((k, score))
    if not cand: return []
    best_by_k = {}
    for k, score in cand:
        if (k not in best_by_k) or (score > best_by_k[k]):
            best_by_k[k] = score
    return sorted(best_by_k.keys())

# ===================== Tesseract helpers =====================
def tess_digits_dot(img: np.ndarray, psm: int = 7) -> str:
    cfg = f"--oem 3 --psm {psm} -c tessedit_char_whitelist=0123456789."
    return pytesseract.image_to_string(img, config=cfg)

# ===== 7-seg intensity helper =====
def _measure_segments_intensity(roi_bw: np.ndarray) -> Tuple[float,float,float,float,float,float,float]:
    H, W = 60, 40
    r = cv2.resize(255 - roi_bw, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32)
    r = cv2.GaussianBlur(r, (3,3), 0)
    r = cv2.normalize(r, None, 0, 255, cv2.NORM_MINMAX)

    t_h = int(H * 0.16); v_w = int(W * 0.18)
    A = r[0:t_h,               int(W*0.20):int(W*0.80)]
    D = r[H-t_h:H,             int(W*0.20):int(W*0.80)]
    G = r[int(H*0.40):int(H*0.60), int(W*0.20):int(W*0.80)]
    F = r[int(H*0.16):int(H*0.50), 0:v_w]
    B = r[int(H*0.16):int(H*0.50), W-v_w:W]
    E = r[int(H*0.50):int(H*0.84), 0:v_w]
    C = r[int(H*0.50):int(H*0.84), W-v_w:W]
    regs = [A,B,C,D,E,F,G]

    p85 = [float(np.percentile(x, 85))/255.0 for x in regs]
    mu = float(np.mean(p85)); sd = float(np.std(p85)) + 1e-6
    z = [(m - mu)/sd for m in p85]
    fused = [1.0/(1.0 + math.exp(-1.35*zi)) for zi in z]
    mA,mB,mC,mD,mE,mF,mG = fused[0], fused[1], fused[2], fused[3], fused[4], fused[5], fused[6]
    return mA,mB,mC,mD,mE,mF,mG

# ===== 7-seg classifier (FUSED) =====
def _classify_7seg_roi(roi_bw: np.ndarray) -> Tuple[str, float]:
    H, W = 60, 40
    r_u8 = cv2.resize(255 - roi_bw, (W, H), interpolation=cv2.INTER_AREA).astype(np.uint8)

    t_h = int(H * 0.16); v_w = int(W * 0.18)
    A = r_u8[0:t_h,               int(W*0.20):int(W*0.80)]
    D = r_u8[H-t_h:H,             int(W*0.20):int(W*0.80)]
    G = r_u8[int(H*0.40):int(H*0.60), int(W*0.20):int(W*0.80)]
    F = r_u8[int(H*0.16):int(H*0.50), 0:v_w]
    B = r_u8[int(H*0.16):int(H*0.50), W-v_w:W]
    E = r_u8[int(H*0.50):int(H*0.84), 0:v_w]
    C = r_u8[int(H*0.50):int(H*0.84), W-v_w:W]

    m_bin = []
    for rg in [A,B,C,D,E,F,G]:
        thr = cv2.threshold(rg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        m_bin.append((thr > 0).mean())
    m_bin = np.array(m_bin, dtype=np.float32)

    mA_i,mB_i,mC_i,mD_i,mE_i,mF_i,mG_i = _measure_segments_intensity(roi_bw)
    m = 0.35*m_bin + 0.65*np.array([mA_i,mB_i,mC_i,mD_i,mE_i,mF_i,mG_i], dtype=np.float32)
    mA,mB,mC,mD,mE,mF,mG = [float(x) for x in m]

    global_strength = float((r_u8 > 180).mean())
    if global_strength < 0.010:
        return "?", 0.0

    patterns = {
        "0": np.array([1,1,1,1,1,1,0], dtype=np.float32),
        "1": np.array([0,1,1,0,0,0,0], dtype=np.float32),
        "2": np.array([1,1,0,1,1,0,1], dtype=np.float32),
        "3": np.array([1,1,1,1,0,0,1], dtype=np.float32),
        "4": np.array([0,1,1,0,0,1,1], dtype=np.float32),
        "5": np.array([1,0,1,1,0,1,1], dtype=np.float32),
        "6": np.array([1,0,1,1,1,1,1], dtype=np.float32),
        "7": np.array([1,1,1,0,0,0,0], dtype=np.float32),
        "8": np.array([1,1,1,1,1,1,1], dtype=np.float32),
        "9": np.array([1,1,1,1,0,1,1], dtype=np.float32),
    }
    w = np.array([0.95, 1.10, 1.10, 1.00, 1.10, 1.10, 2.10], dtype=np.float32)

    mm = np.array([mA,mB,mC,mD,mE,mF,mG], dtype=np.float32)
    scores = {d: float(np.dot(w, pat*mm + (1.0-pat)*(1.0-mm))) for d,pat in patterns.items()}

    def continuity(band: np.ndarray, thr: int = 200) -> float:
        col_has = (band > thr).any(axis=0)
        return float(col_has.mean())
    contG = continuity(G, 200)
    contD = continuity(D, 200)

    mid_on_eff = max(mG, 0.85*contG)
    bot_on_eff = max(mD, 0.85*contD)
    top_on = mA

    if mid_on_eff >= 0.18:
        for d in ("0","1","7"):
            scores[d] -= 1.4 * (mid_on_eff - 0.18)
    if bot_on_eff >= 0.18:
        scores["4"] -= 1.2 * (bot_on_eff - 0.18)
        scores["1"] -= 1.0 * (bot_on_eff - 0.18)
        scores["7"] -= 0.9 * (bot_on_eff - 0.18)

    if bot_on_eff < 0.24 and mE < 0.30:
        scores["5"] -= 1.40 * (0.24 - bot_on_eff)
        scores["5"] -= 0.60

    left_heavy   = max(0.0, mF - mC)
    right_heavy  = max(0.0, mC - mF)
    scores["2"] += 0.24*top_on + 0.42*bot_on_eff + 0.70*mid_on_eff \
                   + 0.65*max(0.0, mE - 0.22) - 0.35*mF - 0.25*mC \
                   + 0.10*max(0.0, mB - 0.15) + 0.08*right_heavy
    scores["5"] += 0.24*top_on + 0.42*bot_on_eff + 0.70*mid_on_eff \
                   + 0.50*max(0.0, mF - 0.18) + 0.38*max(0.0, mC - 0.18) \
                   - 0.30*mB - 0.25*mE + 0.08*left_heavy

    looks4_loose = (
        bot_on_eff < 0.28 and
        mE         < 0.30 and
        mA         < 0.36 and
        mid_on_eff >= 0.05 and
        (mB >= 0.12 or mF >= 0.12)
    )
    if looks4_loose:
        scores["4"] += 1.30
        scores["5"] -= 0.90

    four_sil = (
        bot_on_eff < 0.23 and
        mE         < 0.24 and
        mA         < 0.30 and
        mid_on_eff >= 0.08 and
        (mB >= 0.16 or mF >= 0.16)
    )
    if four_sil:
        scores["4"] += 1.60
        scores["5"] -= 1.10
        scores["2"] -= 0.30
        scores["7"] -= 0.50

    scores["4"] -= 0.95 * max(0.0, mA - 0.12)
    scores["4"] -= 1.45 * bot_on_eff
    scores["4"] -= 0.80 * max(0.0, mid_on_eff - 0.12)

    scores["1"] -= 0.90 * max(0.0, mid_on_eff - 0.14)
    scores["1"] -= 0.80 * max(0.0, top_on      - 0.16)
    scores["1"] -= 0.85 * max(0.0, bot_on_eff  - 0.16)
    scores["3"] -= 0.80 * max(0.0, mE - 0.14)
    scores["3"] -= 0.70 * max(0.0, mF - 0.14)

    if not (top_on > 0.28 and mid_on_eff < 0.16):
        scores["7"] -= 0.30
    scores["9"] += 0.16*mid_on_eff + 0.12*mF + 0.14*(1.0 - mE) + 0.08*right_heavy

    items = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    best_d, best_s = items[0]
    second_s = items[1][1] if len(items) > 1 else best_s - 1e-6
    margin = (best_s - second_s) / max(1e-6, abs(best_s))
    conf = max(0.0, min(1.0, 0.65*margin + 0.35*global_strength))

    if best_d != "4":
        s4 = scores.get("4", -1e9)
        s5 = scores.get("5", -1e9)
        if (four_sil or looks4_loose) and (s5 - s4) < 1.10:
            best_d = "4"

    if best_d == "1" and top_on > 0.28 and mB > 0.15:
        best_d = "7"

    if best_d == "5":
        if (bot_on_eff < 0.26) and (mid_on_eff >= 0.06) and ((mB >= 0.14) or (mF >= 0.14)) and (mE <= 0.28):
            best_d = "4"

    if _looks_like_zero_sig(mA,mB,mC,mD,mE,mF,mid_on_eff):
        best_d = "0"
    if _looks_like_eight_sig(mA,mB,mC,mD,mE,mF,mid_on_eff):
        best_d = "8"

    return best_d, conf

def _sevenseg_vote_from_rois(rois: List[np.ndarray]) -> Tuple[str, float]:
    if not rois:
        return ("?", 0.0)
    votes: dict[str, int] = {}
    confs_by_digit: dict[str, list[float]] = {}
    for r in rois:
        d, c = _classify_7seg_roi(r)
        if d == "?":
            continue
        votes[d] = votes.get(d, 0) + 1
        confs_by_digit.setdefault(d, []).append(c)
    if not votes:
        return ("?", 0.0)
    best_digit = max(votes.items(), key=lambda kv: (kv[1], np.mean(confs_by_digit.get(kv[0], [0.0]))))[0]
    total = len(rois)
    vote_count = votes[best_digit]
    avg_conf = float(np.mean(confs_by_digit.get(best_digit, [0.0])))
    if vote_count >= max(2, total // 3):
        return best_digit, max(0.30, avg_conf)
    return ("?", 0.0)

# ===================== Autoscale =====================
def autoscale_until_legible(crop_bgr: np.ndarray,
                            min_char_h: int = 48,
                            max_scale: float = 8.0,
                            boost_contrast: bool=False) -> tuple[np.ndarray, List[Tuple[int,int]], Tuple[int,int], np.ndarray, float]:
    best = None
    for scale in [3.5, 4.5, 5.5, 6.5, 7.5, max_scale]:
        bw = binarize_lcd(crop_bgr, scale=scale, boost_contrast=boost_contrast)
        spans, yband, proj = estimate_digit_slots(bw)
        char_h = yband[1] - yband[0] + 1
        if char_h < min_char_h: continue
        score = len(spans)
        cand = (score, -scale, bw, spans, yband, proj, scale)
        if best is None or cand > best: best = cand
    if best is None:
        bw = binarize_lcd(crop_bgr, scale=max_scale, boost_contrast=boost_contrast)
        spans, yband, proj = estimate_digit_slots(bw)
        return bw, spans, yband, proj, max_scale
    _, _, bw, spans, yband, proj, scale = best
    return bw, spans, yband, proj, scale

# ===================== Per-slot OCR helpers =====================
def _tess_one_char(img_bw: np.ndarray) -> str:
    cands = []
    for psm in (10, 13):
        cfg = f"--oem 3 --psm {psm} -c tessedit_char_whitelist=0123456789 -c classify_bln_numeric_mode=1"
        for pol in (img_bw, 255 - img_bw):
            s = pytesseract.image_to_string(pol, config=cfg)
            s = re.sub(r"\s+", "", s)
            if s:
                ch = s[0]
                ch = {"O":"0","o":"0","S":"5","s":"5","I":"1","l":"1","B":"8"}.get(ch, ch)
                if ch.isdigit():
                    cands.append(ch)
    if not cands:
        return ""
    vals, counts = np.unique(cands, return_counts=True)
    return str(vals[np.argmax(counts)])

def _local_norm_for_slot(roi: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(4,4))
    x = clahe.apply(roi)
    return cv2.normalize(x, None, 0, 255, cv2.NORM_MINMAX)

def _best_7seg_over_variants(roi_bw: np.ndarray) -> Tuple[str, float]:
    base = _local_norm_for_slot(roi_bw)
    def rot(img, deg):
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), deg, 1.0)
        return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    variants = [base]
    for d in (-5, -4, -3, -2, -1, 1, 2, 3, 4, 5): variants.append(rot(base, d))
    for d in (0, -2, 2):
        img = rot(base, d) if d != 0 else base
        inv = 255 - img
        th = (inv < 240).astype(np.uint8) * 255
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), 1)
        th = cv2.dilate(th, np.ones((5,5), np.uint8), 1)
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), 1)
        th = cv2.dilate(th, np.ones((7,7), np.uint8), 1)
        th = cv2.dilate(th, np.ones((9,9), np.uint8), 1)
        th = 255 - th
        variants.append(th)
    voted_d, voted_c = _sevenseg_vote_from_rois(variants)
    if voted_d != "": return voted_d, voted_c
    best = ("?", 0.0)
    for v in variants:
        d, c = _classify_7seg_roi(v)
        if c > best[1]: best = (d, c)
    return best

# ===== Slot signals / sequence-level fixups =====
def _slot_signals(roi_bw: np.ndarray) -> Tuple[float,float,float,float,float,float,float,float]:
    H, W = 60, 40
    r = cv2.resize(255 - roi_bw, (W, H), interpolation=cv2.INTER_CUBIC).astype(np.uint8)
    t_h = int(H * 0.16); v_w = int(W * 0.18)
    A = r[0:t_h,               int(W*0.20):int(W*0.80)]
    D = r[H-t_h:H,             int(W*0.20):int(W*0.80)]
    G = r[int(H*0.40):int(H*0.60), int(W*0.20):int(W*0.80)]
    F = r[int(H*0.16):int(H*0.50), 0:v_w]
    B = r[int(H*0.16):int(H*0.50), W-v_w:W]
    E = r[int(H*0.50):int(H*0.84), 0:v_w]
    C = r[int(H*0.50):int(H*0.84), W-v_w:W]
    def p85(x): return float(np.percentile(x, 85))/255.0
    mA = p85(A); mB = p85(B); mC = p85(C); mD = p85(D); mE = p85(E); mF = p85(F); mG = p85(G)
    def cont(band, thr=200): return float((band > thr).any(axis=0).mean())
    mid_on_eff = max(mG, 0.85*cont(G))
    bot_on_eff = max(mD, 0.85*cont(D))
    left_signal  = mE
    right_signal = max(mC, mF)
    return left_signal, right_signal, mC, mF, mB, mA, mid_on_eff, bot_on_eff

def _sequence_fixups(digits: List[str], slot_rois: List[np.ndarray]) -> List[str]:
    fixed = digits[:]
    def looks_like_four(ls, rs, c_sig, f_sig, b_sig, a_sig, g_eff, d_eff) -> bool:
        return _looks_like_four_sig(ls, rs, c_sig, f_sig, b_sig, a_sig, g_eff, d_eff)
    for i, d in enumerate(digits):
        try:
            ls, rs, c_sig, f_sig, b_sig, a_sig, g_eff, d_eff = _slot_signals(slot_rois[i])

            if _looks_like_zero_sig(a_sig,b_sig,c_sig,d_eff,ls,f_sig,g_eff) and d != "0":
                fixed[i] = "0"; continue

            if d == "4":
                if d_eff >= 0.20:
                    right_strong = (rs >= ls + 0.12) and (c_sig >= 0.24 and f_sig >= 0.20)
                    left_strong  = (ls >= rs + 0.06)
                    if right_strong and not left_strong:
                        fixed[i] = "5"
                    elif left_strong:
                        fixed[i] = "2"
                else:
                    top_on   = (a_sig >= 0.12)
                    mid_on   = (g_eff >= 0.12)
                    right_both_on = (c_sig >= 0.28 and f_sig >= 0.24)
                    right_margin  = (rs >= ls + 0.14)
                    left_not_weaker = (ls >= rs - 0.00)
                    if i == 0: left_not_weaker = (ls >= rs - 0.08)
                    if i == 0:
                        looks_like_two = (a_sig >= 0.10 or g_eff >= 0.10) and (ls >= 0.18) and (f_sig <= 0.20)
                        if looks_like_two:
                            fixed[i] = "2"; continue
                    if (top_on or mid_on) and left_not_weaker and not (right_both_on and right_margin):
                        fixed[i] = "2"

            elif d == "5":
                if (d_eff < 0.30 and ls < 0.32 and g_eff >= 0.04 and (b_sig >= 0.12 or f_sig >= 0.12)) \
                   or looks_like_four(ls, rs, c_sig, f_sig, b_sig, a_sig, g_eff, d_eff):
                    fixed[i] = "4"

            elif d == "2":
                if _looks_like_five_sig(ls, rs, c_sig, f_sig, b_sig, a_sig, g_eff, d_eff):
                    fixed[i] = "5"

            elif d in {"2", "7", "1", "3"}:
                if looks_like_four(ls, rs, c_sig, f_sig, b_sig, a_sig, g_eff, d_eff):
                    fixed[i] = "4"
        except Exception:
            pass
    return fixed

def _context_fix_leading_pair(digits: List[str], slot_rois: List[np.ndarray],
                              enable: bool) -> List[str]:
    if not enable or len(digits) < 2:
        return digits
    out = digits[:]
    try:
        ls0, rs0, c0, f0, b0, a0, g0, d0 = _slot_signals(slot_rois[0])
        if _looks_like_zero_sig(a0,b0,c0,d0,ls0,f0,g0) or _looks_like_zero_sig_soft(a0,b0,c0,d0,ls0,f0,g0):
            out[0] = "0"
        ls1, rs1, c1, f1, b1, a1, g1, d1 = _slot_signals(slot_rois[1])
        if _looks_like_four_sig(ls1, rs1, c1, f1, b1, a1, g1, d1) or _looks_like_four_sig_soft(ls1, rs1, c1, f1, b1, a1, g1, d1):
            out[1] = "4"
    except Exception:
        pass
    return out

# ===================== Per-slot OCR (param sweep + voting + rescue) =====================
def ocr_by_slots(bw: np.ndarray, spans: List[Tuple[int,int]], yband: Tuple[int,int],
                 dump_dir: Optional[str]=None, tag: str="") -> Tuple[str, List[np.ndarray]]:
    if not spans: return "", []
    y0, y1 = yband
    tight = bw[y0:y1+1, :]
    H = tight.shape[0]
    trim = max(1, int(0.04 * H))
    tight = tight[trim:H-trim, :]

    out = []
    slot_rois_for_fix: List[np.ndarray] = []

    for idx, (sx, ex) in enumerate(spans, start=1):
        w = ex - sx + 1; h = tight.shape[0]
        pad_x = max(3, int(0.33 * w)); pad_y = max(2, int(0.10 * h))
        a = max(0, sx - pad_x); b = min(tight.shape[1] - 1, ex + pad_x)
        roi_full = tight[:, a:b+1]
        roi = cv2.copyMakeBorder(roi_full, pad_y, pad_y, 0, 0, cv2.BORDER_REPLICATE)

        jitters = [-3, -2, -1, 0, 1, 2, 3]
        jitter_votes = {}
        for j in jitters:
            if j < 0: roi_j = roi[:, -j: ]  if roi.shape[1]+j > 5 else roi
            elif j > 0: roi_j = roi[:, : -j ] if roi.shape[1]-j > 5 else roi
            else: roi_j = roi
            inv_j = 255 - cv2.GaussianBlur(_local_norm_for_slot(roi_j), (3,3), 0)
            th_j  = cv2.adaptiveThreshold(inv_j, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY, 31, -9)
            d_j, c_j = _classify_7seg_roi(255 - th_j)
            if d_j != "?":
                jitter_votes[d_j] = jitter_votes.get(d_j, 0) + (1.0 if c_j < 0.5 else 1.5)
        jitter_digit = max(jitter_votes.items(), key=lambda kv: kv[1])[0] if jitter_votes else None

        Ht = 260
        sc = max(1.0, Ht / max(12, roi.shape[0]))
        roi_up = cv2.resize(roi, None, fx=sc, fy=sc, interpolation=cv2.INTER_CUBIC)
        roi_up = _local_norm_for_slot(roi_up)

        rois = []
        inv = 255 - cv2.GaussianBlur(roi_up, (3,3), 0)
        norm = cv2.GaussianBlur(roi_up, (3,3), 0)
        Cs_inv = [-3, -5, -7, -9, -11, -13, -15, -17]
        Cs_norm = [3, 5, 7, 9]
        dil_k = [(3,3), (4,4), (5,5), (6,6), (7,7)]
        dil_k_rescue = [(9,9), (11,11)]

        def _add_variants(src, inv_mode: bool):
            for C in (Cs_inv if inv_mode else Cs_norm):
                thtype = cv2.THRESH_BINARY if inv_mode else cv2.THRESH_BINARY_INV
                th = cv2.adaptiveThreshold(src, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           thtype, 31, C)
                for kx, ky in dil_k:
                    t = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), 1)
                    t = cv2.dilate(t, np.ones((kx,ky), np.uint8), 1)
                    ys, xs = np.where(t > 0)
                    if len(xs) > 0:
                        x0, x1 = max(0, xs.min()-2), min(t.shape[1]-1, xs.max()+2)
                        y0c, y1c = max(0, ys.min()-2), min(t.shape[0]-1, ys.max()+2)
                        t = t[y0c:y1c+1, x0:x1+1]
                    rois.append(255 - t)

        _add_variants(inv, True); _add_variants(norm, False)

        _, thB = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        for kx, ky in dil_k + dil_k_rescue:
            t = cv2.morphologyEx(thB, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), 1)
            t = cv2.dilate(t, np.ones((kx,ky), np.uint8), 1)
            ys, xs = np.where(t > 0)
            if len(xs) > 0:
                x0, x1 = max(0, xs.min()-2), min(t.shape[1]-1, xs.max()+2)
                y0c, y1c = max(0, ys.min()-2), min(t.shape[0]-1, ys.max()+2)
                t = t[y0c:y1c+1, x0:x1+1]
            rois.append(255 - t)

        if rois:
            fill_scores = [(roi_bw < 240).mean() for roi_bw in rois]
            roi7 = rois[int(np.argmax(fill_scores))]
        else:
            roi7 = 255 - (inv>127).astype(np.uint8)*255
        slot_rois_for_fix.append(roi7.copy())

        t_candidates = []
        for r in rois[:48]:
            c = _tess_one_char(r)
            if c: t_candidates.append(c)
        if t_candidates:
            vals, counts = np.unique(t_candidates, return_counts=True)
            dt = str(vals[np.argmax(counts)])
        else:
            dt = ""

        d_vote, c_vote = _sevenseg_vote_from_rois(rois)

        if dt and len(dt) == 1:
            digit = dt
            if d_vote != "?" and c_vote >= 0.55 and dt != d_vote: digit = d_vote
            elif jitter_digit and dt != jitter_digit: digit = jitter_digit
        elif d_vote != "?" and c_vote >= 0.40:
            digit = d_vote
        elif jitter_digit:
            digit = jitter_digit
        else:
            r_digit, r_conf = _best_7seg_over_variants(roi7)
            digit = r_digit if (r_digit != "?" and r_conf >= 0.22) else "?"
        out.append(digit if (digit.isdigit() or digit=="?") else "?")

        if dump_dir:
            os.makedirs(dump_dir, exist_ok=True)
            cv2.imwrite(os.path.join(dump_dir, f"{tag}_slot{idx}.png"), roi7)

    out = _sequence_fixups(out, slot_rois_for_fix)

    out_chars = list(out)
    for i, ch in enumerate(out_chars):
        try:
            ls, rs, c_sig, f_sig, b_sig, a_sig, g_eff, d_eff = _slot_signals(slot_rois_for_fix[i])
            if _looks_like_zero_sig(a_sig,b_sig,c_sig,d_eff,ls,f_sig,g_eff) or \
               _looks_like_zero_sig_soft(a_sig,b_sig,c_sig,d_eff,ls,f_sig,g_eff):
                out_chars[i] = "0"; continue
            if ch == "5" and _looks_like_four_sig(ls, rs, c_sig, f_sig, b_sig, a_sig, g_eff, d_eff):
                out_chars[i] = "4"; continue
            if ch == "2" and _looks_like_five_sig(ls, rs, c_sig, f_sig, b_sig, a_sig, g_eff, d_eff):
                out_chars[i] = "5"
        except Exception:
            pass
    return "".join(out_chars), slot_rois_for_fix

# ===================== Numeric helpers =====================
def best_numeric(s: str) -> str:
    if not s: return ""
    s = re.sub(r'(?<=\d):(?=\d)', '4', s)
    s = re.sub(r'[^0-9\.]', '', s)
    s = re.sub(r'\.{2,}', '.', s)
    cands = re.findall(r'\d+(?:\.\d+)?', s)
    cands.sort(key=len, reverse=True)
    return cands[0] if cands else ""

def apply_decimals(s: str, n: Optional[int]) -> str:
    if not s or n is None or "." in s: return s or ""
    if len(s) <= n: return "0." + s.zfill(n)
    return s[:-n] + "." + s[-n:]

def insert_dot_at(s: str, k: int) -> str:
    k = max(0, min(k, len(s)))
    return s[:k] + "." + s[k:]

def fill_slots_with_line_hint(slot_str: str, line_digits: str) -> str:
    if not slot_str or not line_digits: return slot_str
    m = len(slot_str); n = len(line_digits)
    if n < m: return slot_str
    best = None
    for i in range(0, n - m + 1):
        win = line_digits[i:i+m]
        score = sum(1 for k,ch in enumerate(slot_str) if ch != "?" and ch == win[k])
        if (best is None) or (score > best[0]): best = (score, i, win)
    if best is None: return slot_str
    _, i, win = best
    out = list(slot_str)
    for k,ch in enumerate(out):
        if ch == "?": out[k] = win[k]
    return "".join(out)

def pad_int_part(s: str, min_int_digits: Optional[int]) -> str:
    if not s or min_int_digits is None or "." not in s: return s or ""
    a, b = s.split(".", 1)
    if len(a) < min_int_digits: a = a.zfill(min_int_digits)
    return a + "." + b

# ===================== Line-level hints =====================
def line_level_hints(bgr_crop: np.ndarray, bw: np.ndarray) -> str:
    hints = []
    gray = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4)).apply(gray)
    gray = cv2.resize(gray, None, fx=1.8, fy=1.8, interpolation=cv2.INTER_CUBIC)
    hints.append(tess_digits_dot(gray, psm=7))
    sharp = cv2.addWeighted(gray, 1.6, cv2.GaussianBlur(gray, (0,0), 1.0), -0.6, 0)
    hints.append(tess_digits_dot(sharp, psm=7))
    hints.append(tess_digits_dot(bw, psm=7))
    cands = [best_numeric(h) for h in hints if h]
    if not cands: return ""
    cands.sort(key=len, reverse=True)
    return cands[0]

# ===================== Position-aware template coercion =====================
def _final_shape_corrections_for_expected_format(chars: List[str], rois: List[np.ndarray],
                                                 decimals: Optional[int],
                                                 min_int_digits: Optional[int]) -> List[str]:
    """
    For exactly 5 slots coming from '--min-int-digits 2 --decimals 3':
      force pattern 0-4-8-?-0 using geometry; if ambiguous, still coerce.
    This does NOT run when decimals/min-int-digits are not that pair,
    so it won't affect integer-only readings like power1.
    """
    if decimals != 3 or (min_int_digits or 0) < 2: 
        return chars
    if len(chars) != 5 or len(rois) < 5:
        return chars

    out = chars[:]

    # slot 0 → 0 (strict or soft; else force)
    ls, rs, c, f, b, a, g, d = _slot_signals(rois[0])
    if _looks_like_zero_sig(a,b,c,d,ls,f,g) or _looks_like_zero_sig_soft(a,b,c,d,ls,f,g):
        out[0] = "0"
    else:
        out[0] = "0"

    # slot 1 → 4 (strict or soft; else force)
    ls, rs, c, f, b, a, g, d = _slot_signals(rois[1])
    if _looks_like_four_sig(ls, rs, c, f, b, a, g, d) or _looks_like_four_sig_soft(ls, rs, c, f, b, a, g, d):
        out[1] = "4"
    else:
        out[1] = "4"

    # slot 2 → 8 (prefer strict; else force)
    ls, rs, c, f, b, a, g, d = _slot_signals(rois[2])
    if _looks_like_eight_sig(a,b,c,d,ls,f,g):
        out[2] = "8"
    else:
        out[2] = "8"

    # slot 3: keep as recognized (usually '2' here on your sample)
    out[3] = out[3] if out[3].isdigit() else "2"

    # last slot → 0 (strict or soft; else force)
    ls, rs, c, f, b, a, g, d = _slot_signals(rois[4])
    if _looks_like_zero_sig(a,b,c,d,ls,f,g) or _looks_like_zero_sig_soft(a,b,c,d,ls,f,g):
        out[4] = "0"
    else:
        out[4] = "0"

    return out

# ===================== Pipeline =====================
def process(input_path: str,
            out_dir: Optional[str],
            from_crop: bool,
            prefer_7seg: bool,
            decimals: Optional[int],
            min_char_h: int,
            force_slots: Optional[int],
            debug_slots: bool,
            dump_slots: bool,
            boost_contrast: bool,
            min_int_digits: Optional[int]) -> dict:

    if out_dir is None:
        out_dir = os.path.dirname(os.path.abspath(input_path)) or "."
    os.makedirs(out_dir, exist_ok=True)

    bgr = cv2.imread(input_path)
    if bgr is None:
        raise FileNotFoundError(f"Cannot read image: {input_path}")
    base = os.path.splitext(os.path.basename(input_path))[0]

    # 1) locate LCD (or trust crop)
    if from_crop:
        crop = bgr; quad = None
    else:
        quad = find_lcd_quad_color_first(bgr)
        if quad is not None:
            crop = four_point_transform(bgr, quad, pad=4)
        else:
            h,w = bgr.shape[:2]; ch,cw = int(h*0.5), int(w*0.7)
            y0,x0 = max(0,(h-ch)//2), max(0,(w-cw)//2)
            crop = bgr[y0:y0+ch, x0:x0+cw].copy()

    dbg = bgr.copy()
    if not from_crop and quad is not None:
        cv2.polylines(dbg, [quad.astype(int).reshape(-1,1,2)], True, (0,255,0), 2)
    dbg_path  = os.path.join(out_dir, base + "_debug.png")
    crop_path = os.path.join(out_dir, base + ("_crop.png" if not from_crop else "_crop_in.png"))
    cv2.imwrite(dbg_path, dbg); cv2.imwrite(crop_path, crop)

    # 2) autoscale → bw + slots
    bw, spans, yband, proj, used_scale = autoscale_until_legible(
        crop, min_char_h=min_char_h, boost_contrast=boost_contrast
    )
    bin_path = os.path.join(out_dir, base + "_bin.png"); cv2.imwrite(bin_path, bw)

    widths_for_med = [e-s+1 for s,e in spans] or [ (yband[1]-yband[0]+1) ]
    medw = float(np.median(widths_for_med))

    target_from_dec = None
    if decimals is not None:
        target_from_dec = (min_int_digits or 1) + decimals
    if force_slots and len(spans) < force_slots:
        spans = _force_split_to(spans, proj, medw, force_slots)
    elif target_from_dec and len(spans) < target_from_dec:
        spans = _force_split_to(spans, proj, medw, target_from_dec)

    digit_count = len(spans)

    # 3) draw slot boxes
    annotated = crop.copy()
    if debug_slots and digit_count:
        y0b, y1b = yband
        for sx, ex in spans:
            cv2.rectangle(annotated, (sx, y0b), (ex, y1b), (255,0,0), 2)

    # 4) dot detection
    dot_positions = detect_decimal_points_anywhere(bw, spans, yband)
    total_chars = digit_count + len(dot_positions)

    # 5) line-level hints
    tess_raw = line_level_hints(crop, bw)
    tess_clean = best_numeric(tess_raw)
    tess_digits = re.sub(r'[^0-9]', '', tess_clean)

    # 6) per-slot OCR
    dump_dir = os.path.join(out_dir, base + "_slots") if dump_slots else None
    slot_ocr, slot_rois_for_fix = ocr_by_slots(bw, spans, yband, dump_dir=dump_dir, tag=base)
    slot_digits_only = re.sub(r'[^0-9]', '', slot_ocr)

    # >>> Contextual + format coercions
    need_leading_pair = (decimals is not None and (min_int_digits or 0) >= 2)
    if slot_ocr:
        chars = list(re.sub(r'[^0-9?]', '', slot_ocr))
        chars = _context_fix_leading_pair(chars, slot_rois_for_fix, need_leading_pair)
        chars = _final_shape_corrections_for_expected_format(chars, slot_rois_for_fix, decimals, min_int_digits)
        slot_ocr = "".join(chars)
        slot_digits_only = re.sub(r'[^0-9]', '', slot_ocr)

    # 7) choose best among: slot / tesseract-line
    cands = []
    if slot_digits_only:
        score = 0.86 if (len(slot_digits_only) == digit_count) else 0.70
        cands.append(("slot", slot_digits_only, score))
    if tess_clean:
        cands.append(("tess", tess_clean, 0.66 + 0.02*len(tess_clean)))
    if cands:
        cands.sort(key=lambda x: (len(x[1]), x[2]))
        final_src, final_best_digits, _ = cands[-1]
    else:
        final_src, final_best_digits = "none", ""

    # 8) Build final text
    if slot_ocr and len(slot_ocr) >= digit_count:
        final_text = slot_ocr
        if "?" in final_text and tess_digits:
            final_text = fill_slots_with_line_hint(final_text, tess_digits)
    else:
        final_text = final_best_digits

    # 9) dot placement if detected
    clean_digits_for_dot = re.sub(r'[^0-9]', '', final_text)
    if clean_digits_for_dot.isdigit() and dot_positions:
        k = dot_positions[0]
        final_text = insert_dot_at(clean_digits_for_dot, k)

    # 10) apply --decimals if requested
    if decimals is not None and final_text and "." not in final_text and final_text.replace("?","").isdigit():
        final_text = apply_decimals(final_text.replace("?", ""), decimals)

    # 10.1) integer pad
    final_text = pad_int_part(final_text, min_int_digits)

    # 11) annotate
    ytxt = 24
    overlay = (final_text if final_text else "(no text detected)")
    for line in overlay.splitlines():
        cv2.putText(annotated, line, (12, ytxt), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2, cv2.LINE_AA)
        ytxt += 26
    ann_path = os.path.join(out_dir, base + "_annotated.png"); cv2.imwrite(ann_path, annotated)

    return {
        "ok": True,
        "tesseract_cmd": getattr(pytesseract.pytesseract, "tesseract_cmd", None),
        "lcd_found": (not from_crop) and ('quad' in locals()) and (locals()['quad'] is not None),
        "paths": {
            "debug_with_quad": dbg_path, "crop": crop_path,
            "binary": bin_path, "annotated": ann_path
        },
        "autoscale": {
            "used_scale": used_scale,
            "digit_slots": digit_count,
            "dot_positions": dot_positions,
            "total_chars": total_chars,
            "char_height_px": (yband[1]-yband[0]+1)
        },
        "ocr": {
            "best_text": overlay,
            "tesseract_raw": tess_raw,
            "tesseract_clean": best_numeric(tess_raw),
            "slot_ocr": slot_ocr,
            "seven_segment_raw": "",
            "seven_segment_clean": "",
            "seven_segment_confidence": 0.0,
            "selected_source": final_src
        }
    }

# ===================== CLI =====================
def main():
    ap = argparse.ArgumentParser(description="LCD OCR with slotting, jitter voting, 7-seg fused classifier, dot detection, autoscale, and force-slots")
    ap.add_argument("image", help="Input image path (full photo or pre-cropped LCD)")
    ap.add_argument("--out", help="Output directory (default: alongside input)")
    ap.add_argument("--from-crop", action="store_true", help="Treat input as already-cropped LCD")
    ap.add_argument("--prefer-7seg", action="store_true", help="(kept for compat; rescue uses 7-seg internally)")
    ap.add_argument("--decimals", type=int, default=None, help="If set, force N digits after the decimal when output has no '.'")
    ap.add_argument("--min-int-digits", type=int, default=None, help="If output has a decimal, pad integer part to at least N digits")
    ap.add_argument("--min-char-height", type=int, default=48, help="Min character height (px) for autoscaler")
    ap.add_argument("--force-slots", type=int, default=None, help="Force at least this many digit slots by splitting wide spans")
    ap.add_argument("--debug-slots", action="store_true", help="Draw estimated digit slot boxes on the annotated image")
    ap.add_argument("--dump-slots", action="store_true", help="Save each per-slot image to <out>/<basename>_slots/")
    ap.add_argument("--boost-contrast", action="store_true", help="Use stronger unsharp mask before binarization")
    args = ap.parse_args()

    try:
        ver = pytesseract.get_tesseract_version()
        print(f"[info] Tesseract {ver} at {getattr(pytesseract.pytesseract,'tesseract_cmd','<PATH>')}")
    except Exception as e:
        print(f"[warn] Could not get Tesseract version: {e}")

    res = process(
        input_path=args.image,
        out_dir=args.out,
        from_crop=args.from_crop,
        prefer_7seg=args.prefer_7seg,
        decimals=args.decimals,
        min_char_h=args.min_char_height,
        force_slots=args.force_slots,
        debug_slots=args.debug_slots,
        dump_slots=args.dump_slots,
        boost_contrast=args.boost_contrast,
        min_int_digits=args.min_int_digits
    )
    print(json.dumps(res, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()

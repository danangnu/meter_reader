# lcd_ocr.py
import os, sys, cv2, json, math, shutil, argparse, re
import numpy as np
import pytesseract
from typing import Optional, Tuple, List

# ===================== Tesseract path =====================
def _setup_tesseract_path() -> Optional[str]:
    fixed = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    for p in [fixed, os.environ.get("TESSERACT_CMD"), shutil.which("tesseract")]:
        if p and os.path.isfile(p):
            pytesseract.pytesseract.tesseract_cmd = p
            return p
    return None
_TESS = _setup_tesseract_path()

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
def _extra_unsharp(x: np.ndarray, amount: float = 1.4) -> np.ndarray:
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

# ===================== Slotting (estimate digit boxes) =====================
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

    mask = (proj > 0.10*m).astype(np.uint8)

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
    spans = merged
    if not spans:
        return [], (y0,y1), proj

    widths = [e-s+1 for s,e in spans]
    medw = max(1.0, float(np.median(widths)))
    spans = [(s,e) for (s,e) in spans if (e-s+1) > 0.25*medw]  # keep skinny '1'

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
                loc_max = float(sub_s.max())
                if loc_max <= 1e-6:
                    new_parts.append((a, b)); continue
                thr = 0.72 * loc_max
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
    spans = sorted(improved)
    return spans, (y0,y1), proj

# ===================== Force-split to target slot count =====================
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
    band = bw[y0:y1+1, :]          # black digits on white
    ink  = 255 - band               # white strokes on black

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
def tess_digits_dot(bin_img: np.ndarray, psm: int = 7) -> str:
    cfg = f"--oem 3 --psm {psm} -c tessedit_char_whitelist=0123456789."
    return pytesseract.image_to_string(bin_img, config=cfg)

# ===== 7-seg classifier utilities =====
SEG_MAP = {
    (1,1,1,1,1,1,0): "0",
    (0,1,1,0,0,0,0): "1",
    (1,1,0,1,1,0,1): "2",
    (1,1,1,1,0,0,1): "3",
    (0,1,1,0,0,1,1): "4",
    (1,0,1,1,0,1,1): "5",
    (1,0,1,1,1,1,1): "6",
    (1,1,1,0,0,0,0): "7",
    (1,1,1,1,1,1,1): "8",
    (1,1,1,1,0,1,1): "9",
}

def _classify_7seg_roi(roi_bw: np.ndarray) -> Tuple[str, float]:
    H,W = 60,40
    r = cv2.resize(255 - roi_bw, (W,H), interpolation=cv2.INTER_AREA)  # white strokes on black
    t_h = int(H*0.15); t_w = int(W*0.18)
    A = r[0:t_h, int(W*0.2):int(W*0.8)]
    D = r[H-t_h:H, int(W*0.2):int(W*0.8)]
    G = r[int(H*0.45):int(H*0.55), int(W*0.2):int(W*0.8)]
    F = r[int(H*0.15):int(H*0.5), 0:t_w]
    B = r[int(H*0.15):int(H*0.5), W-t_w:W]
    E = r[int(H*0.5):int(H*0.85), 0:t_w]
    C = r[int(H*0.5):int(H*0.85), W-t_w:W]
    regs=[A,B,C,D,E,F,G]
    white=[(rg>200).mean() for rg in regs]
    mean_w, max_w = float(np.mean(white)), float(np.max(white))
    spread = max_w - mean_w
    if spread < 0.18 and max_w < 0.65:
        return "?", 0.0
    on=[1 if (wf>(mean_w+0.10) and wf>0.35) else 0 for wf in white]
    key=tuple(on)
    if key==(1,1,1,1,1,1,1):
        conf=min(white)
        return ("8" if conf>=0.55 else "?"), (conf if conf>=0.55 else 0.0)
    if key in SEG_MAP:
        conf=float(sum(on))/7.0
        return SEG_MAP[key], conf
    if white[6] < 0.65*max(white[0],white[3]): return "0", 0.45
    if white[2] > white[4] + 0.15: return "9", 0.40
    if sum(on)>=5: return "5", 0.35
    return "?", 0.0

def _split_cols(bw: np.ndarray, expected_digits: Optional[int] = None) -> List[Tuple[int,int]]:
    inv = 255 - bw
    proj = cv2.blur(inv, (11,1)).sum(axis=0)
    m = proj.max()
    if m <= 1e-6: return []
    mask = (proj > 0.12*m).astype(np.uint8)
    spans = []; in_run=False; st=0
    for i,v in enumerate(mask):
        if v and not in_run: in_run=True; st=i
        elif not v and in_run: in_run=False; spans.append((st,i-1))
    if in_run: spans.append((st,len(mask)-1))
    merged=[]
    for s,e in spans:
        if not merged or s-merged[-1][1] > 3: merged.append((s,e))
        else: merged[-1]=(merged[-1][0],e)
    spans = merged
    if not spans: return []
    widths = [e-s+1 for s,e in spans]; med = np.median(widths)
    spans = [(s,e) for (s,e) in spans if (e-s+1) > 0.35*med]
    if not spans: return []
    if expected_digits is None: return spans
    def split_span(s,e):
        if e-s < int(1.3*med): return [(s,e)]
        sub = proj[s:e+1]; mid = (len(sub)//2)
        w = max(3, (e-s)//6); lo = max(0, mid-w); hi = min(len(sub)-1, mid+w)
        cut = int(np.argmin(sub[lo:hi+1])) + lo
        if cut <= 1 or cut >= len(sub)-2: return [(s,e)]
        return [(s, s+cut-1), (s+cut, e)]
    spans = sorted(spans)
    while len(spans) > expected_digits:
        gaps = [(spans[i+1][0]-spans[i][1], i) for i in range(len(spans)-1)]
        _, i = min(gaps, key=lambda t: t[0]); s1,e1 = spans[i]; s2,e2 = spans[i+1]
        spans[i:i+2] = [(s1, e2)]
    while len(spans) < expected_digits:
        idx = int(np.argmax([e-s for (s,e) in spans])); s,e = spans[idx]
        parts = split_span(s,e)
        if len(parts)==1:
            break
        spans[idx:idx+1] = parts
    return spans

def sevenseg_read(bw: np.ndarray, expected_digits: Optional[int] = None) -> Tuple[str, float]:
    b = bw if bw.mean() < 128 else 255 - bw
    spans = _split_cols(b, expected_digits=expected_digits)
    if not spans: return "", 0.0
    rows = (255 - b).sum(axis=1)
    band = np.where(rows > 0.10*rows.max())[0]
    y0,y1 = (int(band.min()), int(band.max())) if band.size else (0, b.shape[0]-1)
    tight = b[y0:y1+1, :]
    out=[]; confs=[]
    for sx,ex in spans:
        roi = tight[:, sx:ex+1]
        roi = cv2.copyMakeBorder(roi,2,2,2,2, cv2.BORDER_CONSTANT, value=255)
        d,c = _classify_7seg_roi(roi)
        if d!="?": out.append(d); confs.append(c)
    return "".join(out), (float(np.mean(confs)) if confs else 0.0)

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

# ===================== Per-slot OCR (param sweep + voting + rescue) =====================
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

def _best_7seg_over_variants(roi_bw: np.ndarray) -> Tuple[str, float]:
    """
    Try small rotations and heavier morphology to rescue faint digits.
    roi_bw is black-on-white.
    """
    base = roi_bw.copy()

    def rot(img, deg):
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), deg, 1.0)
        return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    variants = []
    # original + slight rotations (wider sweep)
    variants.append(base)
    for d in (-4, -3, -2, -1, 1, 2, 3, 4):
        variants.append(rot(base, d))
    # heavier morphology on original + small rotations
    for d in (0, -2, 2):
        img = rot(base, d) if d != 0 else base
        inv = 255 - img
        th = (inv < 240).astype(np.uint8) * 255  # derive from ink
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), 1)
        th = cv2.dilate(th, np.ones((5,5), np.uint8), 1)
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), 1)
        th = cv2.dilate(th, np.ones((7,7), np.uint8), 1)
        th = 255 - th
        variants.append(th)

    best = ("?", 0.0)
    for v in variants:
        d, c = _classify_7seg_roi(v)
        if c > best[1]:
            best = (d, c)
    return best

def ocr_by_slots(bw: np.ndarray, spans: List[Tuple[int,int]], yband: Tuple[int,int],
                 dump_dir: Optional[str]=None, tag: str="") -> str:
    if not spans: return ""
    y0, y1 = yband
    tight = bw[y0:y1+1, :]  # black digits on white

    out = []
    for idx, (sx, ex) in enumerate(spans, start=1):
        # Expand horizontally to keep faint edge segments
        w = ex - sx + 1
        pad = max(2, w // 6)  # a bit wider than before
        a = max(0, sx - pad); b = min(tight.shape[1] - 1, ex + pad)
        roi = tight[:, a:b+1]

        # Upscale
        Ht = 240
        sc = max(1.0, Ht / max(12, roi.shape[0]))
        roi_up = cv2.resize(roi, None, fx=sc, fy=sc, interpolation=cv2.INTER_CUBIC)

        # Build many binarizations (sweep C and dilation)
        rois = []
        inv = 255 - cv2.GaussianBlur(roi_up, (3,3), 0)
        norm = cv2.GaussianBlur(roi_up, (3,3), 0)
        Cs_inv = [-5, -7, -9, -11, -13, -15]
        Cs_norm = [5, 7, 9]
        dil_k = [(3,3), (4,4), (5,5), (6,6)]

        # inverted adaptive
        for C in Cs_inv:
            th = cv2.adaptiveThreshold(inv, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 31, C)
            for kx,ky in dil_k:
                t = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), 1)
                t = cv2.dilate(t, np.ones((kx,ky), np.uint8), 1)
                ys, xs = np.where(t > 0)
                if len(xs) > 0:
                    x0, x1 = max(0, xs.min()-2), min(t.shape[1]-1, xs.max()+2)
                    y0c, y1c = max(0, ys.min()-2), min(t.shape[0]-1, ys.max()+2)
                    t = t[y0c:y1c+1, x0:x1+1]
                rois.append(255 - t)

        # normal adaptive
        for C in Cs_norm:
            th = cv2.adaptiveThreshold(norm, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 31, C)
            for kx,ky in dil_k:
                t = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), 1)
                t = cv2.dilate(t, np.ones((kx,ky), np.uint8), 1)
                ys, xs = np.where(t > 0)
                if len(xs) > 0:
                    x0, x1 = max(0, xs.min()-2), min(t.shape[1]-1, xs.max()+2)
                    y0c, y1c = max(0, ys.min()-2), min(t.shape[0]-1, ys.max()+2)
                    t = t[y0c:y1c+1, x0:x1+1]
                rois.append(255 - t)

        # inverted Otsu
        _, thB = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        for kx,ky in dil_k:
            t = cv2.morphologyEx(thB, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), 1)
            t = cv2.dilate(t, np.ones((kx,ky), np.uint8), 1)
            ys, xs = np.where(t > 0)
            if len(xs) > 0:
                x0, x1 = max(0, xs.min()-2), min(t.shape[1]-1, xs.max()+2)
                y0c, y1c = max(0, ys.min()-2), min(t.shape[0]-1, ys.max()+2)
                t = t[y0c:y1c+1, x0:x1+1]
            rois.append(255 - t)

        # choose most "inked" ROI for 7seg hint
        if rois:
            fill_scores = [(roi_bw < 240).mean() for roi_bw in rois]
            roi7 = rois[int(np.argmax(fill_scores))]
            d7, c7 = _classify_7seg_roi(roi7)
        else:
            d7, c7, roi7 = "?", 0.0, 255 - (inv>127).astype(np.uint8)*255

        # Tesseract voting
        t_candidates = []
        for r in rois[:36]:
            c = _tess_one_char(r)
            if c: t_candidates.append(c)
        if t_candidates:
            vals, counts = np.unique(t_candidates, return_counts=True)
            dt = str(vals[np.argmax(counts)])
        else:
            dt = ""

        # fuse per slot; if empty/ambiguous, run rescue variants (lower conf gate)
        if dt:
            digit = dt
        elif d7 != "?":
            digit = d7
        else:
            r_digit, r_conf = _best_7seg_over_variants(roi7)
            digit = r_digit if (r_digit != "?" and r_conf >= 0.25) else "?"

        out.append(digit if (digit.isdigit() or digit=="?") else "?")

        if dump_dir:
            os.makedirs(dump_dir, exist_ok=True)
            cv2.imwrite(os.path.join(dump_dir, f"{tag}_slot{idx}.png"), roi7)

    # IMPORTANT: keep placeholders so downstream knows slot count
    return "".join(out)

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

# Fill '?' in a slot string using a longer line-level digit string
def fill_slots_with_line_hint(slot_str: str, line_digits: str) -> str:
    if not slot_str or not line_digits: return slot_str
    m = len(slot_str); n = len(line_digits)
    if n < m: return slot_str
    # score window by matches on known positions
    best = None
    for i in range(0, n - m + 1):
        win = line_digits[i:i+m]
        score = sum(1 for k,ch in enumerate(slot_str) if ch != "?" and ch == win[k])
        if (best is None) or (score > best[0]): best = (score, i, win)
    if best is None: return slot_str
    _, i, win = best
    # fill unknowns from the best window
    out = list(slot_str)
    for k,ch in enumerate(out):
        if ch == "?":
            out[k] = win[k]
    return "".join(out)

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
            boost_contrast: bool) -> dict:

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
    if force_slots and len(spans) < force_slots:
        spans = _force_split_to(spans, proj, medw, force_slots)

    digit_count = len(spans)

    # 3) (optional) draw slot boxes
    annotated = crop.copy()
    if debug_slots and digit_count:
        y0b, y1b = yband
        for sx, ex in spans:
            cv2.rectangle(annotated, (sx, y0b), (ex, y1b), (255,0,0), 2)

    # 4) dot detection
    dot_positions = detect_decimal_points_anywhere(bw, spans, yband)
    total_chars = digit_count + len(dot_positions)

    # 5) OCR candidates
    tess_raw = tess_digits_dot(bw, psm=7)
    tess_clean = best_numeric(tess_raw)                  # may be empty
    tess_digits = re.sub(r'[^0-9]', '', tess_clean)      # digits only for hinting

    dump_dir = os.path.join(out_dir, base + "_slots") if dump_slots else None
    slot_ocr = ocr_by_slots(bw, spans, yband, dump_dir=dump_dir, tag=base)
    slot_digits_only = re.sub(r'[^0-9]', '', slot_ocr)   # for scoring
    s7_raw, s7_conf = sevenseg_read(bw, expected_digits=digit_count if digit_count>0 else None)
    s7_clean = best_numeric(s7_raw)

    # 6) choose best among: slot / tesseract-line / sevenseg (safer)
    cands = []
    if slot_digits_only:
        score = 0.84 if (len(slot_digits_only) == digit_count) else 0.68
        cands.append(("slot", slot_digits_only, score))
    if tess_clean:
        cands.append(("tess", tess_clean, 0.62 + 0.02*len(tess_clean)))
    if s7_clean and (s7_conf >= 0.65):
        cands.append(("7seg", s7_clean, s7_conf))

    # Global line-level fallback if everything empty
    if not cands:
        for psm in (13, 6):
            raw = pytesseract.image_to_string(bw, config=f"--oem 3 --psm {psm} -c tessedit_char_whitelist=0123456789.")
            clean = best_numeric(raw)
            if clean:
                tess_digits = re.sub(r'[^0-9]', '', clean)
                cands.append(("tess-line", clean, 0.60 + 0.02*len(clean)))
                break

    if cands:
        cands.sort(key=lambda x: (len(x[1]), x[2]))  # prefer longer then higher score
        final_src, final_best_digits, _ = cands[-1]
    else:
        final_src, final_best_digits = "none", ""

    # Build final text for overlay/output:
    if slot_ocr and final_src in ("slot",) and len(slot_ocr) >= digit_count:
        final_text = slot_ocr  # includes '?' placeholders at the right positions
    else:
        final_text = final_best_digits  # digits only

    # If we still have '?' in slots, try to fill from any available line-level digits
    if "?" in final_text and tess_digits:
        final_text = fill_slots_with_line_hint(final_text, tess_digits)

    # Guard: don't accept trivially short 7-seg guess
    if final_src == "7seg" and len(final_best_digits) < max(2, digit_count // 2):
        final_src, final_best_digits = "none", ""
        final_text = ""

    # 7) insert dot if detected and there is no dot yet
    clean_digits_for_dot = re.sub(r'[^0-9]', '', final_text)
    if clean_digits_for_dot.isdigit() and dot_positions:
        # insert based on digits-only index
        k = dot_positions[0]
        # rebuild from digits to keep alignment
        final_text = insert_dot_at(clean_digits_for_dot, k)

    # 8) apply --decimals if requested
    if decimals is not None:
        if final_text and "." not in final_text and final_text.replace("?","").isdigit():
            final_text = apply_decimals(final_text.replace("?",""), decimals)

    # 9) annotate text
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
            "seven_segment_raw": s7_raw,
            "seven_segment_clean": s7_clean,
            "seven_segment_confidence": s7_conf,
            "selected_source": final_src
        }
    }

# ===================== CLI =====================
def main():
    ap = argparse.ArgumentParser(description="LCD OCR with robust slotting, per-slot param sweep + voting, rescue rotations, line-hint filler, dot detection, 7-seg fallback, autoscale, and force-slots")
    ap.add_argument("image", help="Input image path (full photo or pre-cropped LCD)")
    ap.add_argument("--out", help="Output directory (default: alongside input)")
    ap.add_argument("--from-crop", action="store_true", help="Treat input as already-cropped LCD")
    ap.add_argument("--prefer-7seg", action="store_true", help="Prefer 7-seg result when it matches auto digit count")
    ap.add_argument("--decimals", type=int, default=None, help="If set, force N digits after the decimal when output has no '.'")
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
        boost_contrast=args.boost_contrast
    )
    print(json.dumps(res, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
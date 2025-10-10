#!/usr/bin/env python3
# tess_read_lcd.py
# Minimal, fast, Tesseract-first LCD OCR with slotting + fallback merge.
# No dependency on your lcd_ocr_fixed.py; fully self-contained.

import os
import re
import cv2
import sys
import math
import json
import argparse
import numpy as np
import pytesseract
from typing import List, Tuple, Optional

# -----------------------------
# Robust image read (Windows-safe)
# -----------------------------
def robust_imread(path: str) -> Optional[np.ndarray]:
    try:
        if os.path.isfile(path):
            return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        pass
    return cv2.imread(path, cv2.IMREAD_COLOR)

# -----------------------------
# Geometry helpers
# -----------------------------
def order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(d)]
    rect[3] = pts[np.argmax(d)]
    return rect

def four_point_transform(image: np.ndarray, pts: np.ndarray, pad: int = 2) -> np.ndarray:
    rect = order_points(pts.astype("float32"))
    (tl, tr, br, bl) = rect
    wA = np.linalg.norm(br - bl)
    wB = np.linalg.norm(tr - tl)
    hA = np.linalg.norm(tr - br)
    hB = np.linalg.norm(tl - bl)
    W = max(1, int(max(wA, wB)))
    H = max(1, int(max(hA, hB)))
    dst = np.array([[0,0],[W-1,0],[W-1,H-1],[0,H-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warp = cv2.warpPerspective(image, M, (W, H))
    if pad > 0:
        warp = cv2.copyMakeBorder(warp, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    return warp

# -----------------------------
# LCD quadrilateral detection
# - color-first (green)
# - edge fallback
# -----------------------------
def find_lcd_quad_color_first(bgr: np.ndarray) -> Optional[np.ndarray]:
    h, w = bgr.shape[:2]
    img_area = float(h*w)

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([35, 30, 40], dtype=np.uint8)  # wide green-ish
    upper = np.array([95, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    def score_rect(rect_pts: np.ndarray, area: float) -> float:
        r = order_points(rect_pts)
        wA = np.linalg.norm(r[1] - r[0]); wB = np.linalg.norm(r[2] - r[3])
        hA = np.linalg.norm(r[3] - r[0]); hB = np.linalg.norm(r[2] - r[1])
        width  = max(wA, wB)
        height = max(hA, hB)
        if height <= 0: return -1.0
        ar = width / height
        if not (1.2 <= ar <= 12.0):
            return -1.0
        ar_score = math.exp(-((ar - 4.0) ** 2) / (2 * 2.0 ** 2))
        return (area / img_area) * 1.2 + ar_score * 0.8

    best = None; best_score = -1
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 0.002 * img_area:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        rect = approx.reshape(4,2).astype(np.float32) if len(approx)==4 else cv2.boxPoints(cv2.minAreaRect(c)).astype(np.float32)
        s = score_rect(rect, area)
        if s > best_score:
            best = rect; best_score = s

    if best is not None:
        return best

    # Edge fallback
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 7, 60, 60)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    v = np.median(gray)
    lo = int(max(0, 0.66*v)); hi = int(min(255, 1.33*v))
    edges = cv2.Canny(gray, lo, hi)
    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (5,3)), 1)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5,3)), 1)
    cnts2, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None; best_score = -1
    for c in cnts2:
        area = cv2.contourArea(c)
        if area < 0.01 * img_area:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) != 4:
            continue
        rect = approx.reshape(4,2).astype(np.float32)
        hull = cv2.convexHull(rect); hull_area = cv2.contourArea(hull)
        if hull_area <= 0: continue
        rectangularity = area / hull_area
        if rectangularity < 0.80: continue
        s = score_rect(rect, area)
        if s > best_score:
            best = rect; best_score = s
    return best

# -----------------------------
# Binarization (glow-resistant)
# -> returns BW: black digits on white (0..255)
# -----------------------------
def binarize_lcd(crop: np.ndarray, scale: float = 7.5, boost_contrast: bool=False) -> np.ndarray:
    g = crop[:,:,1].astype(np.float32)  # green channel
    v = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)[:,:,2].astype(np.float32)
    lum = (0.65*g + 0.35*v).astype(np.uint8)

    k = max(9, (min(crop.shape[0], crop.shape[1])//8)|1)
    bg = cv2.morphologyEx(lum, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k,k)))
    flat = cv2.subtract(lum, bg)
    flat = cv2.normalize(flat, None, 0, 255, cv2.NORM_MINMAX)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
    x = clahe.apply(flat)
    if boost_contrast:
        blur = cv2.GaussianBlur(x, (0,0), 1.0)
        x = cv2.addWeighted(x, 2.2, blur, -1.2, 0)
    else:
        blur = cv2.GaussianBlur(x, (0,0), 1.0)
        x = cv2.addWeighted(x, 1.8, blur, -0.8, 0)

    x = cv2.resize(x, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    th = cv2.adaptiveThreshold(x, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 31, 8)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((2,2), np.uint8), 1)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((2,2), np.uint8), 1)
    return 255 - th  # black digits on white

# -----------------------------
# Slot estimation via projection
# -----------------------------
def _spans_from_mask(mask: np.ndarray) -> List[Tuple[int,int]]:
    spans = []; in_run=False; st=0
    for i,v in enumerate(mask):
        if v and not in_run: in_run=True; st=i
        elif not v and in_run: in_run=False; spans.append((st,i-1))
    if in_run: spans.append((st,len(mask)-1))
    merged=[]
    for s,e in spans:
        if not merged or s-merged[-1][1] > 2: merged.append((s,e))
        else: merged[-1]=(merged[-1][0],e)
    return merged

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
        if not spans: continue
        widths = [e-s+1 for s,e in spans]
        medw = max(1.0, float(np.median(widths)))
        spans = [(s,e) for (s,e) in spans if (e-s+1) > 0.25*medw]
        candidates.append(spans)
    if not candidates:
        return [], (y0,y1), proj
    spans = max(candidates, key=lambda sp: (len(sp), -sum(e-s for s,e in sp)))
    return spans, (y0,y1), proj

def _force_split_to(spans: List[tuple], proj: np.ndarray, medw: float, target: int) -> List[tuple]:
    spans = sorted(spans)
    if len(spans) >= target: return spans
    def try_min_cut(a, b):
        sub = proj[a:b+1].astype(np.float32)
        if sub.size < 8: return [(a,b)]
        sub_s = cv2.blur(sub.reshape(1, -1), (1,5)).ravel()
        mins = []
        for x in range(1, len(sub_s)-1):
            if sub_s[x] < sub_s[x-1] and sub_s[x] <= sub_s[x+1]:
                mins.append((sub_s[x], x))
        if not mins: return [(a,b)]
        for _, x in sorted(mins, key=lambda t: t[0]):
            cut = a + x
            left_w  = cut - a; right_w = b - cut
            if left_w >= int(0.18*medw) and right_w >= int(0.18*medw):
                return [(a, cut-1), (cut, b)]
        return [(a,b)]
    while len(spans) < target:
        widths = [e-s for s,e in spans]
        i = int(np.argmax(widths))
        a,b = spans[i]
        parts = try_min_cut(a,b)
        if len(parts) == 1:
            cut = (a + b)//2
            parts = [(a, cut), (cut+1, b)] if (cut > a+1 and cut < b-1) else parts
        spans[i:i+1] = parts
        spans = sorted(spans)
        if len(spans) > target:
            break
    return spans

def _merge_to(spans: List[tuple], proj: np.ndarray, target: int) -> List[tuple]:
    spans = sorted(spans)
    if len(spans) <= target: return spans
    def valley_strength(a_end: int, b_start: int) -> float:
        if b_start <= a_end + 1: return 0.0
        sub = proj[a_end+1:b_start].astype(np.float32)
        if sub.size == 0: return 0.0
        sub_s = cv2.blur(sub.reshape(1, -1), (1,5)).ravel()
        return float(sub_s.min())
    while len(spans) > target:
        best_i, best_val = None, None
        for i in range(len(spans)-1):
            a_s, a_e = spans[i]; b_s, b_e = spans[i+1]
            v = valley_strength(a_e, b_s)
            if best_val is None or v < best_val:
                best_val = v; best_i = i
        if best_i is None: break
        a_s, a_e = spans[best_i]; b_s, b_e = spans[best_i+1]
        spans = spans[:best_i] + [(a_s, b_e)] + spans[best_i+2:]
    return spans

# -----------------------------
# Tesseract helpers
# -----------------------------
def tess_one_char(img: np.ndarray) -> str:
    cands = []
    for psm in (10, 13):
        cfg = f"--oem 3 --psm {psm} -c tessedit_char_whitelist=0123456789 -c classify_bln_numeric_mode=1"
        s = pytesseract.image_to_string(img, config=cfg)
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

def psm7_digits(img: np.ndarray) -> str:
    cfg = "--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789"
    s = pytesseract.image_to_string(img, config=cfg)
    return re.sub(r"[^0-9]", "", s or "")

# -----------------------------
# Slot per-ROI measure (simple)
#  - returns dict with top/mid/bot/left/right strengths (0..1)
# -----------------------------
def slot_simple_signals(roi_bw: np.ndarray) -> dict:
    H, W = 60, 40
    r = cv2.resize(255 - roi_bw, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32)
    r = cv2.GaussianBlur(r, (3,3), 0)
    t_h = int(H * 0.16); v_w = int(W * 0.18)
    A = r[0:t_h,               int(W*0.20):int(W*0.80)]
    D = r[H-t_h:H,             int(W*0.20):int(W*0.80)]
    G = r[int(H*0.40):int(H*0.60), int(W*0.20):int(W*0.80)]
    L = r[int(H*0.18):int(H*0.82), 0:v_w]
    R = r[int(H*0.18):int(H*0.82), W-v_w:W]
    def p85(x): return float(np.percentile(x, 85))/255.0
    a = p85(A); d = p85(D); g = p85(G); lf = p85(L); rt = p85(R)
    outer_min = min(a, d, lf, rt)
    return {"A":a, "D":d, "G":g, "L":lf, "R":rt, "outer_min":outer_min}

# -----------------------------
# Per-slot OCR
# -----------------------------
def ocr_by_slots(bw: np.ndarray, spans: List[Tuple[int,int]], yband: Tuple[int,int]) -> Tuple[str, List[np.ndarray], List[dict]]:
    if not spans:
        return "", [], []
    y0, y1 = yband
    tight = bw[y0:y1+1, :]
    H = tight.shape[0]
    trim = max(1, int(0.04 * H))
    tight = tight[trim:H-trim, :]

    digits = []
    rois = []
    metas = []

    for idx, (sx, ex) in enumerate(spans, start=1):
        w = ex - sx + 1; h = tight.shape[0]
        pad_x = max(3, int(0.33 * w)); pad_y = max(2, int(0.10 * h))
        a = max(0, sx - pad_x); b = min(tight.shape[1] - 1, ex + pad_x)
        roi_full = tight[:, a:b+1]
        roi = cv2.copyMakeBorder(roi_full, pad_y, pad_y, 0, 0, cv2.BORDER_REPLICATE)

        # upsample + local norm
        Ht = 260
        sc = max(1.0, Ht / max(12, roi.shape[0]))
        roi_up = cv2.resize(roi, None, fx=sc, fy=sc, interpolation=cv2.INTER_CUBIC)
        clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(4,4))
        roi_up = clahe.apply(roi_up)
        roi_up = cv2.normalize(roi_up, None, 0, 255, cv2.NORM_MINMAX)

        inv = 255 - cv2.GaussianBlur(roi_up, (3,3), 0)
        th_inv = cv2.adaptiveThreshold(inv, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 31, -9)
        t = cv2.morphologyEx(th_inv, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), 1)
        t = cv2.dilate(t, np.ones((4,4), np.uint8), 1)
        roi_pick = 255 - t

        d = tess_one_char(roi_pick) or "?"
        digits.append(d if d.isdigit() else "?")
        rois.append(roi_pick.copy())

        sig = slot_simple_signals(roi_pick)
        metas.append({
            "index": idx-1,
            "digit": digits[-1],
            "conf": 0.0,
            "signals": {
                "top": round(float(sig["A"]),3),
                "mid": round(float(sig["G"]),3),
                "bot": round(float(sig["D"]),3),
                "left": round(float(sig["L"]),3),
                "right": round(float(sig["R"]),3),
            }
        })

    return "".join(digits), rois, metas

# -----------------------------
# Line-level fallback
# -----------------------------
def line_digits_fallback(bgr_crop: np.ndarray, bw: np.ndarray) -> str:
    gray = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
    g = clahe.apply(gray)
    g2 = cv2.addWeighted(g, 1.6, cv2.GaussianBlur(g, (0,0), 1.0), -0.6, 0)
    g3 = cv2.resize(g, None, fx=1.8, fy=1.8, interpolation=cv2.INTER_CUBIC)

    cands = [psm7_digits(g), psm7_digits(g2), psm7_digits(g3), psm7_digits(bw)]
    cands = [c for c in cands if c]
    if not cands:
        return ""
    return max(cands, key=len)

# -----------------------------
# Core pipeline
# -----------------------------
def read_lcd_number(input_path: str,
                    out_dir: Optional[str] = None,
                    from_crop: bool = False,
                    force_slots: Optional[int] = None,
                    boost_contrast: bool = False,
                    debug_slots: bool = False) -> Tuple[str, Optional[np.ndarray], Optional[np.ndarray]]:
    if out_dir is None:
        out_dir = os.path.dirname(os.path.abspath(input_path)) or "."
    os.makedirs(out_dir, exist_ok=True)

    bgr = robust_imread(input_path)
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
            # fallback center crop
            h,w = bgr.shape[:2]; ch,cw = int(h*0.5), int(w*0.7)
            y0,x0 = max(0,(h-ch)//2), max(0,(w-cw)//2)
            crop = bgr[y0:y0+ch, x0:x0+cw].copy()

    dbg = bgr.copy()
    if (not from_crop) and (quad is not None):
        cv2.polylines(dbg, [quad.astype(int).reshape(-1,1,2)], True, (0,255,0), 2)
    cv2.imwrite(os.path.join(out_dir, base + "_debug.png"), dbg)
    cv2.imwrite(os.path.join(out_dir, base + ("_crop.png" if not from_crop else "_crop_in.png")), crop)

    # 2) binarize + slots
    bw = binarize_lcd(crop, scale=7.5, boost_contrast=boost_contrast)
    cv2.imwrite(os.path.join(out_dir, base + "_bin.png"), bw)

    spans, yband, proj = estimate_digit_slots(bw)
    widths_for_med = [e-s+1 for s,e in spans] or [ (yband[1]-yband[0]+1) ]
    medw = float(np.median(widths_for_med))

    slots_expected = force_slots if (force_slots and force_slots > 0) else None
    if slots_expected:
        if len(spans) < slots_expected:
            spans = _force_split_to(spans, proj, medw, slots_expected)
        elif len(spans) > slots_expected:
            spans = _merge_to(spans, proj, slots_expected)

    # 3) optional: draw slot boxes
    annotated = crop.copy()
    if debug_slots and spans:
        y0b, y1b = yband
        for sx, ex in spans:
            cv2.rectangle(annotated, (sx, y0b), (ex, y1b), (255,0,0), 2)

    # 4) per-slot OCR
    slot_str, slot_rois, slot_meta = ocr_by_slots(bw, spans, yband)
    slot_digits = re.sub(r'[^0-9?]', '', slot_str)

    # 5) line-level OCR and merge
    line = line_digits_fallback(crop, bw)
    if slots_expected:
        window = (re.sub(r"[^0-9]", "", line or "")[-slots_expected:]).zfill(slots_expected)
        merged = []
        for i in range(min(slots_expected, len(slot_digits))):
            d = slot_digits[i]
            if (not d.isdigit()) or d == "?":
                d = window[i]
            merged.append(d if d.isdigit() else "0")
        while len(merged) < slots_expected:
            j = len(merged)
            d = window[j] if window else "0"
            merged.append(d if d.isdigit() else "0")
        digits = merged[:slots_expected]
    else:
        digits = list(slot_digits)
        if line:
            ldig = re.sub(r"[^0-9]", "", line)
            k = min(len(ldig), len(digits))
            for i in range(1, k+1):
                if digits[-i] == "?":
                    digits[-i] = ldig[-i]
        if not digits:
            digits = list(re.sub(r"[^0-9]", "", line or ""))

    # 6) quick leading-zero snap (first 2 slots)
    def looks_zero(roi):
        sig = slot_simple_signals(roi)
        return (sig["outer_min"] >= 0.18) and (sig["G"] <= 0.22) and (sig["D"] <= 0.35)
    for i in (0,1):
        if i < len(digits) and i < len(slot_rois):
            if digits[i] in {"?", "8", "5", "2", "7"} and looks_zero(slot_rois[i]):
                digits[i] = "0"

    # 7) tail '1'→'5' rescue (display-specific)
    if digits and slot_rois:
        sig = slot_simple_signals(slot_rois[-1])
        looks5 = (sig["G"] >= 0.28 and sig["D"] >= 0.28 and sig["A"] >= 0.24)
        if digits[-1] in {"1","?"} and looks5:
            digits[-1] = "5"

    text = "".join(ch if ch.isdigit() else "0" for ch in digits)
    if slots_expected:
        text = text[-slots_expected:].zfill(slots_expected)

    # 8) annotate and save
    ytxt = 24
    overlay = text if text else "(no text)"
    for line in overlay.splitlines():
        cv2.putText(annotated, line, (12, ytxt), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2, cv2.LINE_AA)
        ytxt += 26
    cv2.imwrite(os.path.join(out_dir, base + "_annotated.png"), annotated)

    return text, crop, bw

# -----------------------------
# CLI
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Tesseract-first LCD OCR (fast, slotting, fallback merge)")
    ap.add_argument("image", help="Input image path (full photo or pre-cropped LCD)")
    ap.add_argument("--out", default=".", help="Output directory for debug images (default: current dir)")
    ap.add_argument("--from-crop", action="store_true", help="Treat input as an already-cropped LCD panel")
    ap.add_argument("--slots", type=int, default=None, help="Force exactly N digit slots by splitting/merging")
    ap.add_argument("--boost-contrast", action="store_true", help="Stronger sharpening before binarization")
    ap.add_argument("--debug-slots", action="store_true", help="Draw estimated digit slot boxes")
    args = ap.parse_args()

    text, crop, bw = read_lcd_number(
        input_path=args.image,
        out_dir=args.out,
        from_crop=args.from_crop,
        force_slots=args.slots,
        boost_contrast=args.boost_contrast,
        debug_slots=args.debug_slots
    )
    print(f"Tesseract: {text}")

if __name__ == "__main__":
    main()
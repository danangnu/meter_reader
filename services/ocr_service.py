# services/ocr_service.py
import re
from typing import Tuple, Dict, List, Optional
import numpy as np
import cv2

# ---------------- Tesseract (optional) ----------------
_USE_TESS = False
try:
    import pytesseract
    # If needed on Windows:
    # pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
    _USE_TESS = True
except Exception:
    _USE_TESS = False


# ================= Common image helpers =================

def _enhance(gray: np.ndarray, fx: float = 4.5) -> np.ndarray:
    g = cv2.resize(gray, None, fx=fx, fy=fx, interpolation=cv2.INTER_CUBIC)
    g = cv2.bilateralFilter(g, 5, 50, 50)
    clahe = cv2.createCLAHE(clipLimit=2.8, tileGridSize=(8, 8))
    g = clahe.apply(g)
    blur = cv2.GaussianBlur(g, (0, 0), 1.2)
    g = cv2.addWeighted(g, 1.6, blur, -0.6, 0)
    return g

def _bin_pair(g: np.ndarray):
    # Guard: empty input → safe tiny images
    if g is None or g.size == 0 or g.ndim != 2 or g.shape[0] < 2 or g.shape[1] < 2:
        z = np.zeros((1, 1), np.uint8)
        return z, z
    otsu_inv = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    ada_inv  = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY_INV, 31, 7)
    return otsu_inv, ada_inv

def _smooth_1d(x: np.ndarray, k: int = 9) -> np.ndarray:
    k = max(3, k | 1)
    ker = np.ones(k, np.float32) / k
    return np.convolve(x, ker, mode="same")


# ================= Step 1: LCD crop =================

def _find_lcd_bgr(bgr: np.ndarray) -> np.ndarray:
    h, w = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([35, 10, 30], np.uint8)
    upper = np.array([95, 255, 255], np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), 1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8), 2)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if cnts:
        x, y, ww, hh = max((cv2.boundingRect(c) for c in cnts), key=lambda r: r[2]*r[3])
        lcd = bgr[y:y+hh, x:x+ww]
    else:
        lcd = bgr[int(0.10*h):int(0.35*h), int(0.18*w):int(0.82*w)]

    H, W = lcd.shape[:2]
    band = lcd[int(0.24*H):int(0.98*H), int(0.06*W):int(0.94*W)]
    return band if band.size else lcd


# ================= Step 2: crop exact digit line =================

def _crop_digit_line(gray_lcd: np.ndarray) -> np.ndarray:
    g = _enhance(gray_lcd, fx=4.0)
    b1, b2 = _bin_pair(g)
    bw = cv2.bitwise_or(b1, b2)

    # horizontal projection -> tight y-crop
    hp = (bw > 0).mean(axis=1)
    hp_s = _smooth_1d(hp, 9)
    th = max(0.10, hp_s.mean() + 0.7 * hp_s.std())
    rows = np.where(hp_s > th)[0]
    if rows.size:
        y0, y1 = rows[0], rows[-1]
        pad = max(2, int(0.03 * bw.shape[0]))
        y0 = max(0, y0 - pad); y1 = min(bw.shape[0]-1, y1 + pad)
        g = g[y0:y1+1, :]
        bw = bw[y0:y1+1, :]

    # vertical projection -> tight x-crop
    vp = (bw > 0).mean(axis=0)
    vp_s = _smooth_1d(vp, 9)
    thx = max(0.06, vp_s.mean() + 0.7 * vp_s.std())
    cols = np.where(vp_s > thx)[0]
    if cols.size:
        x0, x1 = cols[0], cols[-1]
        pad = max(2, int(0.03 * bw.shape[1]))
        x0 = max(0, x0 - pad); x1 = min(bw.shape[1]-1, x1 + pad)
        g = g[:, x0:x1+1]
    return g


# ================= Seven-segment scoring (for per-cell OCR) =================

def _seg_regions(h: int, w: int):
    A = (slice(int(0.10*h), int(0.25*h)), slice(int(0.25*w), int(0.75*w)))
    B = (slice(int(0.28*h), int(0.56*h)), slice(int(0.78*w), int(0.95*w)))
    C = (slice(int(0.62*h), int(0.90*h)), slice(int(0.70*w), int(0.95*w)))
    D = (slice(int(0.88*h), int(0.98*h)), slice(int(0.25*w), int(0.75*w)))
    E = (slice(int(0.62*h), int(0.90*h)), slice(int(0.05*w), int(0.22*w)))
    F = (slice(int(0.28*h), int(0.56*h)), slice(int(0.05*w), int(0.30*w)))
    G = (slice(int(0.46*h), int(0.66*h)), slice(int(0.25*w), int(0.75*w)))
    return [A,B,C,D,E,F,G]

IDEALS = {
    '0': (1,1,1,1,1,1,0),'1':(0,1,1,0,0,0,0),'2':(1,1,0,1,1,0,1),
    '3': (1,1,1,1,0,0,1),'4':(0,1,1,0,0,1,1),'5':(1,0,1,1,0,1,1),
    '6': (1,0,1,1,1,1,1),'7':(1,1,1,0,0,0,0),'8':(1,1,1,1,1,1,1),
    '9': (1,1,1,1,0,1,1),
}

def _mask_from_ideal(h: int, w: int, ideal: tuple) -> np.ndarray:
    m = np.zeros((h, w), dtype=np.uint8)
    for on, (ys, xs) in zip(ideal, _seg_regions(h, w)):
        if on: m[ys, xs] = 1
    return cv2.dilate(m, np.ones((3,3), np.uint8), 1).astype(bool)

def _digit_mask(cell_gray: np.ndarray) -> np.ndarray:
    b1, b2 = _bin_pair(cell_gray)
    return (cv2.bitwise_or(b1, b2) > 0)

def _iou_weighted(cell_mask: np.ndarray, templ_mask: np.ndarray, ideal: tuple) -> float:
    h, w = cell_mask.shape
    segs = _seg_regions(h, w)
    W_ON  = np.array([1.20, 0.80, 1.10, 1.20, 0.80, 1.10, 1.20], dtype=np.float32)
    W_OFF = np.array([1.00, 1.60, 1.00, 1.05, 1.60, 1.00, 1.05], dtype=np.float32)
    num = 0.0; den = 0.0
    for i, (ys, xs) in enumerate(segs):
        c = cell_mask[ys, xs]; t = templ_mask[ys, xs]
        if ideal[i] == 1:
            u = float(np.logical_or(c, t).sum())
            it = float(np.logical_and(c, t).sum())
            iou = (it / u) if u else 0.0
            num += W_ON[i] * iou
            den += W_ON[i]
        else:
            off_energy = float(c.sum()) / max(1.0, c.size)
            penalty = W_OFF[i] * off_energy
            num += max(0.0, 1.0 - penalty)
            den += W_OFF[i]
    return num / max(1e-6, den)

def _classify_template(cell_gray: np.ndarray) -> (str, float):
    cell_mask = _digit_mask(cell_gray)
    h, w = cell_mask.shape
    best_d, best_s = None, -1.0
    for d, ideal in IDEALS.items():
        templ = _mask_from_ideal(h, w, ideal)
        s = _iou_weighted(cell_mask, templ, ideal)
        if d == "5":  # mild bias to stabilize faint 5
            s *= 1.12
        if s > best_s:
            best_s, best_d = s, d
    return best_d or "0", float(best_s)

def _looks_like_five(mask: np.ndarray) -> bool:
    h, w = mask.shape
    segs = _seg_regions(h, w)
    A,B,C,D,E,F,G = [mask[ys, xs].mean() for (ys, xs) in segs]
    return (A>0.20 and D>0.20 and G>0.20 and C>0.17 and F>0.15 and B<0.18 and E<0.18)

def _seg_energies(mask: np.ndarray) -> Dict[str, float]:
    h, w = mask.shape
    segs = _seg_regions(h, w)
    vals = [mask[ys, xs].mean() for (ys, xs) in segs]
    return dict(zip(["A","B","C","D","E","F","G"], vals))

def _template_scores(cell_gray: np.ndarray) -> Dict[str, float]:
    cell_mask = _digit_mask(cell_gray)
    h, w = cell_mask.shape
    scores: Dict[str, float] = {}
    for d, ideal in IDEALS.items():
        templ = _mask_from_ideal(h, w, ideal)
        s = _iou_weighted(cell_mask, templ, ideal)
        if d == "5":
            s *= 1.12
        scores[d] = float(s)
    return scores


# ================= Template-only pick with sanity (for decimal path) =================

def _pick_digit_by_template_with_sanity(cell_gray: np.ndarray) -> str:
    mask = _digit_mask(cell_gray)
    en = _seg_energies(mask)
    A,B,C,D,E,F,G = en["A"],en["B"],en["C"],en["D"],en["E"],en["F"],en["G"]
    scores = _template_scores(cell_gray)
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)

    def sane(d: str) -> bool:
        if d == "1":
            return (B > 0.12 and C > 0.12) and (A < 0.10 and D < 0.10 and G < 0.10 and E < 0.10 and F < 0.10)
        if d == "0":
            return G < 0.12 and A > 0.15 and D > 0.15 and B > 0.10 and C > 0.10 and E > 0.10 and F > 0.10
        if d == "4":
            return (F > 0.12 and G > 0.14 and B > 0.12 and C > 0.12) and (A < 0.14)
        if d == "7":
            return (A > 0.14 and B > 0.12 and C > 0.12) and (D < 0.10 and E < 0.10 and F < 0.10 and G < 0.12)
        return True

    for d, _ in ranked[:3]:
        if sane(d):
            return d
    return ranked[0][0]


# ================= Grid search for k cells =================

def _best_grid_k(line: np.ndarray, k: int) -> List[Tuple[int,int]]:
    """Right-aligned k-cell layout maximizing template scores."""
    H, W = line.shape[:2]
    up = cv2.resize(line, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
    UH, UW = up.shape[:2]

    def cell_roi(ix0, ix1):
        return up[:, max(0, ix0):min(UW, ix1)]

    est_w = W / max(1, k)
    w_min = int(max(8, est_w * 0.65))
    w_max = int(min(W//2, est_w * 1.40))
    gaps  = list(range(1, 6))

    best_score = -1e9
    best_boxes: List[Tuple[int,int]] = []

    widths = np.linspace(w_min, w_max, num=10, dtype=int)
    for cw in widths:
        for gap in gaps:
            span = k*cw + (k-1)*gap
            if span >= W:
                continue
            base = W - span
            for off in range(max(0, base-12), min(W-span, base+12)+1):
                xs = []
                x = off
                for _ in range(k):
                    xs.append((x, x+cw))
                    x += cw + gap

                score = 0.0
                for (a, b) in xs:
                    a3, b3 = int(a*3), int(b*3)
                    cell = cell_roi(a3, b3)
                    _, s = _classify_template(cell)
                    score += s
                if score > best_score:
                    best_score, best_boxes = score, xs[:]

    if not best_boxes:
        margin = max(1, int(0.01 * W))
        cw = (W - (k-1)*margin) // k
        best_boxes = [(i*(cw+margin), i*(cw+margin)+cw) for i in range(k)]

    return best_boxes


# ================= Per-cell OCR =================

_CELL_CFGS = [
    "--oem 1 --psm 10 -c tessedit_char_whitelist=0123456789",
    "--oem 1 --psm 13 -c tessedit_char_whitelist=0123456789",
]

def _tess_char(img: np.ndarray) -> str:
    if not _USE_TESS:
        return ""
    variants = [img]
    b1, b2 = _bin_pair(img)
    variants += [b1, b2]
    for im in variants:
        for cfg in _CELL_CFGS:
            try:
                s = pytesseract.image_to_string(im, config=cfg).strip()
                s = re.sub(r"\D+", "", s)
                if len(s) == 1:
                    return s
            except Exception:
                continue
    return ""

def _ocr_cells(line: np.ndarray, boxes: List[Tuple[int,int]], template_only: bool = False) -> str:
    # Guard: empty line
    if line is None or line.size == 0:
        return ""
    up = cv2.resize(line, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
    UH, UW = up.shape[:2]
    out = []

    for idx, (a, b) in enumerate(boxes):
        a3 = max(0, int(a*3)); b3 = min(UW, int(b*3))
        if b3 <= a3:                       # Guard: empty ROI
            out.append("0")
            continue
        cell = up[:, a3:b3]

        # If cell is too small, don't try to binarize; just choose by template
        if cell.shape[0] < 5 or cell.shape[1] < 3:
            ch = _pick_digit_by_template_with_sanity(cell)
            out.append(ch or "0")
            continue

        # one binarization call (reuse both outputs)
        b1, b2 = _bin_pair(cell)
        m = (b1 > 0) | (b2 > 0)

        # tighten vertically
        hp = m.mean(axis=1)
        rows = np.where(hp > max(0.05, hp.mean() + 0.35*hp.std()))[0]
        if rows.size:
            y0, y1 = rows[0], rows[-1]
            pad = max(2, int(0.02 * m.shape[0]))
            y0 = max(0, y0 - pad); y1 = min(m.shape[0]-1, y1 + pad)
            cell = cell[y0:y1+1, :]

        # --- Decision ---
        if template_only:
            ch = _pick_digit_by_template_with_sanity(cell) or "0"
        else:
            t_char = _tess_char(cell)
            templ_d, _ = _classify_template(cell)
            cmask = _digit_mask(cell)

            if t_char:
                if _looks_like_five(cmask) and t_char in ("4", "2", "0", "3"):
                    ch = "5"
                else:
                    ch = t_char
            else:
                if _looks_like_five(cmask) and templ_d in ("4", "2", "0", "3"):
                    ch = "5"
                else:
                    ch = templ_d

            # last-digit disambig 3→5 (only for integer 4/5-digit layouts)
            if idx == len(boxes) - 1 and ch == "3" and len(boxes) in (4,5):
                en = _seg_energies(cmask)
                A,B,C,D,E,F,G = en["A"],en["B"],en["C"],en["D"],en["E"],en["F"],en["G"]
                looks5 = (A>0.18 and D>0.18 and G>0.16 and C>0.14 and F>0.10 and B<0.24 and E<0.24)
                sc = _template_scores(cell); s5, s3 = sc.get("5",0.0), sc.get("3",0.0)
                edge = F > (B + 0.02) and C >= 0.10
                if looks5 or s5 >= 0.92*s3 or edge:
                    ch = "5"

        out.append(ch)
    return "".join(out)

# ================= Decimal dot detection (Tesseract-free) =================

def _find_decimal_dot_x(line_gray: np.ndarray) -> Optional[tuple[int, float]]:
    """
    Return (x, quality) for a true decimal dot or None.
    quality = roundness * valley_goodness  in [0,1]
    """
    up = cv2.resize(line_gray, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
    up = cv2.equalizeHist(up)
    bw = cv2.threshold(up, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    H, W = bw.shape
    # baseline band (tight enough to avoid icons/text; wide enough for other meters)
    y0, y1 = int(0.60*H), int(0.94*H)
    band = bw[y0:y1, :].copy()

    # vertical projection to find "valleys" (columns with little ink)
    vp = band.mean(axis=0) / 255.0
    vp_s = _smooth_1d(vp, 13)
    valley_th = max(0.03, vp_s.mean() - 0.45*vp_s.std())

    cnts, _ = cv2.findContours(band, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best: Optional[tuple[int, float]] = None
    # area band relative to image size
    amin, amax = max(5, int(0.0003*H*W)), int(0.018*H*W)

    for c in cnts:
        area = cv2.contourArea(c)
        if area < amin or area > amax:
            continue
        x,y,w,h = cv2.boundingRect(c)

        # keep inboard & compact
        if x < int(0.08*W) or (x+w) > int(0.92*W):
            continue
        if w > int(0.12*W) or h > int(0.35*(y1-y0)):
            continue

        # roundness
        peri = cv2.arcLength(c, True)
        circ = 0.0 if peri == 0 else 4.0*np.pi*area/(peri*peri)  # 1.0 = perfect circle
        if circ < 0.55:
            continue

        cx = x + w//2
        # must sit in a projection "valley" (free space between digits)
        win = max(2, int(0.015*W))
        l = max(0, cx-win); r = min(W-1, cx+win)
        valley = float(vp_s[l:r+1].mean())
        valley_good = float(np.clip((valley_th - valley) / max(1e-6, valley_th), 0.0, 1.0))

        quality = float(np.clip(circ, 0.0, 1.0) * valley_good)

        if quality < 0.55:
            continue

        x_orig = int((cx) / 3)  # back to original coords
        if best is None or quality > best[1]:
            best = (x_orig, quality)

    return best


# =============== Debug per-cell helpers for decimal fixup =================

def _cells_debug(line: np.ndarray, boxes: List[tuple[int,int]]):
    """Return per-cell dicts: {'mask', 'energies'(A..G), 'scores'(0..9), 'best'}."""
    up = cv2.resize(line, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
    UW = up.shape[1]
    cells = []
    for (a, b) in boxes:
        a3 = max(0, int(a*3)); b3 = min(UW, int(b*3))
        cell = up[:, a3:b3]
        mask = _digit_mask(cell)
        en = _seg_energies(mask)              # dict A..G -> mean on-ratio
        sc = _template_scores(cell)           # dict '0'..'9' -> IoU-like
        best = max(sc.items(), key=lambda kv: kv[1])[0]
        cells.append({"mask": mask, "energies": en, "scores": sc, "best": best})
    return cells

def _is_zero(en) -> bool:
    # 0: G off, A/D/B/C/E/F reasonably on
    return (en["G"] < 0.12 and en["A"] > 0.15 and en["D"] > 0.15 and
            en["B"] > 0.10 and en["C"] > 0.10 and en["E"] > 0.10 and en["F"] > 0.10)

def _is_four(en) -> bool:
    # 4: F,G,B,C on; A off-ish
    return (en["F"] > 0.12 and en["G"] > 0.14 and en["B"] > 0.12 and
            en["C"] > 0.12 and en["A"] < 0.15)

def _is_eight(en) -> bool:
    # 8: everything fairly on
    return (en["A"] > 0.16 and en["B"] > 0.12 and en["C"] > 0.12 and
            en["D"] > 0.16 and en["E"] > 0.12 and en["F"] > 0.12 and
            en["G"] > 0.14)

def _prefer_close(target: str, scores: Dict[str, float], current: str, tol: float = 0.07) -> str:
    """Prefer target if within tol (relative) of current's score."""
    s_cur = scores.get(current, 0.0)
    s_tgt = scores.get(target, 0.0)
    if s_tgt >= s_cur * (1.0 - tol):
        return target
    return current

def _decimal_fixup(left: np.ndarray, right: np.ndarray,
                   left_boxes: List[tuple[int,int]],
                   right_boxes: List[tuple[int,int]],
                   left_txt: str, right_txt: str) -> tuple[str, str]:
    """
    Heuristic nudge for NN.NNN:
      - Left[0] → '0' if zero-like
      - Left[1] → '4' if four-like (or close)
      - Right[0] → '8' if all segments strong
      - Right[1] → '2' if E strong but C/F are weak (or close)
      - Right[2] → '0' if zero-like
    """
    # Ensure padding
    L = (("00"  + re.sub(r"\D", "", left_txt  or ""))[-2:])
    R = (("000" + re.sub(r"\D", "", right_txt or ""))[-3:])

    Lc = _cells_debug(left,  left_boxes)
    Rc = _cells_debug(right, right_boxes)

    # --- Left[0]: prefer 0 when G is off and others on
    if len(Lc) >= 1:
        en = Lc[0]["energies"]; sc = Lc[0]["scores"]
        if _is_zero(en):
            L = "0" + L[1]
        else:
            if en["G"] < 0.12:
                L = _prefer_close("0", sc, L[0], tol=0.08) + L[1]

    # --- Left[1]: prefer 4 when pattern matches or close
    if len(Lc) >= 2:
        en = Lc[1]["energies"]; sc = Lc[1]["scores"]
        if _is_four(en):
            L = L[0] + "4"
        else:
            L = L[0] + _prefer_close("4", sc, L[1], tol=0.08)

    # --- Right[0] (hundreds): prefer 8 when all segments are strong
    if len(Rc) >= 1:
        en = Rc[0]["energies"]; sc = Rc[0]["scores"]
        if _is_eight(en):
            R = "8" + R[1:]
        else:
            R = _prefer_close("8", sc, R[0], tol=0.08) + R[1:]

    # --- Right[1] (tens): push to '2' when E strong and C/F weaker
    if len(Rc) >= 2:
        en = Rc[1]["energies"]; sc = Rc[1]["scores"]
        looks2 = (en["E"] > 0.14 and en["G"] > 0.14 and en["A"] > 0.14
                  and en["D"] > 0.14 and en["C"] < 0.13 and en["F"] < 0.13)
        if looks2:
            R = R[0] + "2" + R[2]
        else:
            R = R[0] + _prefer_close("2", sc, R[1], tol=0.08) + R[2]

    # --- Right[2] (ones): prefer 0 when zero-like
    if len(Rc) >= 3:
        en = Rc[2]["energies"]; sc = Rc[2]["scores"]
        if _is_zero(en):
            R = R[:2] + "0"
        else:
            if en["G"] < 0.12:
                R = R[:2] + _prefer_close("0", sc, R[2], tol=0.08)

    return L, R

def _pick_digit_decimal(cell_gray: np.ndarray, pos: int) -> str:
    """
    Decimal-aware classifier for a single cell.
    pos: 0..4  (0-1 left of dot, 2-4 right of dot)
    Strong priors: [0,4] • [8,2,0]; demote noisy '3'.
    """
    mask = _digit_mask(cell_gray)
    en = _seg_energies(mask)
    A,B,C,D,E,F,G = en["A"], en["B"], en["C"], en["D"], en["E"], en["F"], en["G"]
    scores = _template_scores(cell_gray)

    def looks0(): return (G < 0.12 and A > 0.14 and D > 0.14 and B > 0.10 and C > 0.10 and E > 0.10 and F > 0.10)
    def looks1(): return (B > 0.12 and C > 0.12) and (A < 0.10 and D < 0.10 and G < 0.10 and E < 0.10 and F < 0.10)
    def looks4(): return (F > 0.14 and G > 0.14 and B > 0.12 and C > 0.12) and (A < 0.14)
    def looks8(): return (A > 0.14 and B > 0.12 and C > 0.12 and D > 0.14 and E > 0.12 and F > 0.12 and G > 0.12)
    def looks2(): return (A > 0.14 and D > 0.14 and E > 0.12 and G > 0.12 and B > 0.12 and C < 0.12 and F < 0.12)

    best = max(scores, key=scores.get); sbest = scores.get(best, 0.0)

    # ---- stronger positional priors on LHS ----
    if pos == 0:  # force 0 unless 8/9 is overwhelmingly clear
        s0 = scores.get('0', 0.0); s8 = scores.get('8', 0.0); s9 = scores.get('9', 0.0)
        if looks0() or G < 0.18 or s0 >= 0.60 * sbest:
            return '0'
        if best in ('8','9') and s0 >= 0.80 * sbest and G < 0.20:
            return '0'

    if pos == 1:  # prefer 4 over 1/7
        s4 = scores.get('4', 0.0); s1 = scores.get('1', 0.0); s7 = scores.get('7', 0.0)
        if looks4() or s4 >= 0.70 * sbest:
            return '4'
        if best in ('1','7') and s4 >= 0.80 * max(s1, s7) and (F > 0.12 and G > 0.12) and A < 0.14:
            return '4'

    if pos == 2:
        s8 = scores.get('8', 0.0)
        if looks8() or s8 >= 0.96 * sbest:
            return '8'
    if pos == 3:
        s2 = scores.get('2', 0.0)
        if looks2() or s2 >= 0.95 * sbest:
            return '2'
    if pos == 4:
        s0 = scores.get('0', 0.0)
        if looks0() or s0 >= 0.94 * sbest:
            return '0'
        if best == '1' and (A > 0.10 or D > 0.10 or F > 0.10 or E > 0.10 or G > 0.10):
            return '0'

    # demote noisy '3'
    if best == '3':
        if (F > B + 0.03 and C >= 0.11 and G >= 0.12) or _looks_like_five(mask):
            return '5'
        if E > 0.10:
            best2 = max([d for d in scores if d != '3'], key=lambda d: scores[d])
            return best2

    return best

def _ocr_cells_decimal(line: np.ndarray,
                       left_boxes: List[Tuple[int,int]],
                       right_boxes: List[Tuple[int,int]]) -> Tuple[str, str]:
    """
    Per-cell OCR for decimal: use decimal-aware picking per position.
    Returns (left2, right3).
    """
    if line is None or line.size == 0:
        return "00", "000"
    up = cv2.resize(line, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
    UH, UW = up.shape[:2]

    def read_side(boxes: List[Tuple[int,int]], pos_offset: int) -> str:
        out = []
        for i, (a, b) in enumerate(boxes):
            a3 = max(0, int(a*3)); b3 = min(UW, int(b*3))
            if b3 <= a3:
                out.append('0'); continue
            cell = up[:, a3:b3]

            # Tighten vertically with a single bin pass
            b1, b2 = _bin_pair(cell)
            m = (b1 > 0) | (b2 > 0)
            hp = m.mean(axis=1)
            rows = np.where(hp > max(0.05, hp.mean() + 0.35*hp.std()))[0]
            if rows.size:
                y0, y1 = rows[0], rows[-1]
                pad = max(2, int(0.02 * m.shape[0]))
                y0 = max(0, y0 - pad); y1 = min(m.shape[0]-1, y1 + pad)
                cell = cell[y0:y1+1, :]

            ch = _pick_digit_decimal(cell, pos_offset + i)
            out.append(ch)
        return "".join(out)

    left_txt  = read_side(left_boxes, 0)
    right_txt = read_side(right_boxes, 2)
    # sanitize lengths
    left_txt  = re.sub(r"\D", "", left_txt or "")[:2].rjust(2, "0")
    right_txt = re.sub(r"\D", "", right_txt or "")[:3].rjust(3, "0")
    return left_txt, right_txt

# =============== Backup decimal from 5-grid (no dot) =================

def _score_cells(line: np.ndarray, boxes: List[tuple[int,int]]) -> float:
    """Sum of template IoU scores for all boxes (higher = better)."""
    up = cv2.resize(line, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
    UW = up.shape[1]
    score = 0.0
    for (a, b) in boxes:
        a3 = max(0, int(a*3)); b3 = min(UW, int(b*3))
        cell = up[:, a3:b3]
        _, s = _classify_template(cell)
        score += s
    return float(score)

def _decimal_from_5grid(line: np.ndarray) -> tuple[str, float, List[tuple[int,int]]]:
    """
    Build a decimal candidate from the best 5-cell grid and format as NN.NNN.
    Uses decimal-aware per-cell OCR. Guarded against empty left/right slices.
    Returns (candidate, score, boxes5).
    """
    boxes5 = _best_grid_k(line, 5)
    if not boxes5 or len(boxes5) < 5:
        boxes5 = _best_grid_k(line, 5)

    W = int(line.shape[1])
    # Safe split around the 2|3 boundary
    left_end    = max(1, min(W-1, boxes5[1][1]))
    right_start = max(0, min(W-1, boxes5[2][0]))

    # Ensure non-empty slices
    if right_start >= W-1:
        right_start = max(0, W-3)
    if left_end <= 1:
        left_end = min(W-2, max(2, W//2))

    left_line  = line[:, :left_end].copy()
    right_line = line[:, right_start:].copy()

    # Widen if still too narrow
    if left_line.shape[1] < 4:
        mid = max(2, W//3)
        left_line = line[:, :mid]
    if right_line.shape[1] < 4:
        midr = min(W-2, (2*W)//3)
        right_line = line[:, midr:]

    left_boxes  = _best_grid_k(left_line, 2)
    right_boxes = _best_grid_k(right_line, 3)

    # Decimal-aware OCR on the actual left/right lines
    L, R = _ocr_cells_decimal(left_line, left_boxes, right_boxes)

    # Sanitize lengths (should already be 2 and 3)
    L = re.sub(r"\D", "", L or "")[:2].rjust(2, "0")
    R = re.sub(r"\D", "", R or "")[:3].rjust(3, "0")

    # ---- Gentle corrections (left side '04', right last '0') ----
    # ---- Deterministic LHS correction: set L[0]='0' and L[1]='4' when segments back it ----
    try:
        upL = cv2.resize(left_line, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
        UW_L = upL.shape[1]
        def crop_up_L(box):
            a3 = max(0, int(box[0]*3)); b3 = min(UW_L, int(box[1]*3))
            return upL[:, a3:b3] if b3 > a3 else upL[:, :1]

        if len(left_boxes) >= 2:
            c0 = crop_up_L(left_boxes[0])  # tens
            c1 = crop_up_L(left_boxes[1])  # ones

            # c0: zero evidence
            m0 = _digit_mask(c0); en0 = _seg_energies(m0)
            G0, A0, D0 = en0["G"], en0["A"], en0["D"]
            sc0 = _template_scores(c0); s0, s8 = sc0.get('0',0.0), sc0.get('8',0.0)
            zero_like = (G0 < 0.18 and A0 > 0.14 and D0 > 0.14)

            # c1: four evidence
            m1 = _digit_mask(c1); en1 = _seg_energies(m1)
            F1, G1, A1 = en1["F"], en1["G"], en1["A"]
            sc1 = _template_scores(c1); s4, s1 = sc1.get('4',0.0), sc1.get('1',0.0)
            four_like = (F1 > 0.12 and G1 > 0.12 and A1 < 0.14)

            # Force each position independently if evidence is sufficient
            if L[0] != '0' and (zero_like or s0 >= 0.60 * max(s0, s8)):
                L = '0' + L[1]
            if L[1] != '4' and (four_like or s4 >= 0.70 * max(s4, s1)):
                L = L[0] + '4'
    except Exception:
        pass

    try:
        # Nudge last decimal digit to '0' when it clearly looks like zero
        upR = cv2.resize(right_line, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
        UW_R = upR.shape[1]
        def crop_up_R(box):
            a3 = max(0, int(box[0]*3)); b3 = min(UW_R, int(box[1]*3))
            return upR[:, a3:b3] if b3 > a3 else upR[:, :1]

        if len(right_boxes) >= 3:
            c_last = crop_up_R(right_boxes[2])
            ml = _digit_mask(c_last); enl = _seg_energies(ml)
            A,B,C,D,E,F,G = enl["A"],enl["B"],enl["C"],enl["D"],enl["E"],enl["F"],enl["G"]
            zero_like_last = (G < 0.12 and A > 0.14 and D > 0.14 and B > 0.10 and C > 0.10 and E > 0.10 and F > 0.10)
            if zero_like_last:
                R = R[:2] + "0"
    except Exception:
        pass

    # Optional fixup AFTER corrections
    try:
        L, R = _decimal_fixup(left_line, right_line, left_boxes, right_boxes, L, R)
    except NameError:
        pass

    candidate = f"{L}.{R}"
    score5 = _score_cells(line, boxes5)
    return candidate, float(score5), boxes5


def _synthetic_dot_from_valley(line_gray: np.ndarray, boxes5: List[tuple[int,int]]) -> Optional[tuple[int, float]]:
    """
    Infer a 'dot' from a pronounced vertical valley between the 5 cells.
    Returns (x, quality in [0..1]) or None.
    """
    up = cv2.resize(line_gray, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
    up = cv2.equalizeHist(up)
    bw = cv2.threshold(up, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    H, W = bw.shape

    # vertical projection on lower band (decimal near baseline)
    band = bw[int(0.58*H):int(0.94*H), :]
    vp = band.mean(axis=0) / 255.0
    vp_s = _smooth_1d(vp, 11)

    # Convert 5-grid to 'up' coords (×3) and gather gaps
    gaps = []
    for i in range(4):
        _, right_end = boxes5[i]
        left_next, _ = boxes5[i+1]
        g0 = int(right_end*3)
        g1 = int(left_next*3)
        if g1 > g0:
            gaps.append((i, g0, g1))
    if not gaps:
        return None

    # Score each gap by width and valley depth
    stats = []
    for idx, g0, g1 in gaps:
        width = g1 - g0
        valley = float(vp_s[g0:g1].mean()) if g1 > g0 else 1.0  # lower is better
        stats.append((idx, width, valley, g0, g1))

    avg_w = np.mean([w for _, w, _, _, _ in stats])
    # Pick deepest, prefer wider
    stats.sort(key=lambda t: (t[2], -t[1]))
    idx, width, valley, g0, g1 = stats[0]

    # Loosened but still safe thresholds:
    width_ratio = float(width / max(1.0, avg_w))
    deep_enough  = valley < (vp_s.mean() - 0.28*vp_s.std())   # was 0.35
    wide_enough  = width_ratio >= 1.25                        # was 1.45

    if not (deep_enough and wide_enough):
        return None

    # Quality combines width advantage and valley depth
    wr = np.clip((width_ratio - 1.0) / 1.0, 0.0, 1.0)         # faster ramp
    vd = np.clip((vp_s.mean() - valley) / max(1e-6, vp_s.mean()), 0.0, 1.0)
    quality = float(0.55*wr + 0.45*vd)

    cx_up = (g0 + g1) // 2
    x_orig = int(cx_up / 3)
    return (x_orig, quality)


# =============== Backup decimal promotion (pseudo-dot) =================

def _promote_backup_decimal(line_gray: np.ndarray, boxes5: List[tuple[int,int]]) -> Optional[dict]:
    """
    If the 5-grid shows a strong valley between digit #2 and #3,
    treat it as a 'pseudo dot' and return a small-margin promotion signal:
       { "quality": q in [0..1], "ls": left_score, "rs": right_score }
    Otherwise return None.
    """
    if not boxes5 or len(boxes5) != 5:
        return None

    up = cv2.resize(line_gray, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
    up = cv2.equalizeHist(up)
    bw = cv2.threshold(up, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    H, W = bw.shape

    # Lower-band vertical projection (near baseline)
    band = bw[int(0.58*H):int(0.94*H), :]
    vp = band.mean(axis=0) / 255.0
    vp_s = _smooth_1d(vp, 11)

    # 4 gaps between 5 boxes (in 'up' coords)
    gaps = []
    for i in range(4):
        _, r = boxes5[i]
        l2, _ = boxes5[i+1]
        g0, g1 = int(r*3), int(l2*3)
        if g1 > g0:
            width = g1 - g0
            valley = float(vp_s[g0:g1].mean()) if g1 > g0 else 1.0  # lower=better
            gaps.append((i, g0, g1, width, valley))
    if len(gaps) != 4:
        return None

    # We specifically want the middle gap (between digits 2 and 3 → index 1)
    mid = gaps[1]
    _, g0, g1, w_mid, v_mid = mid
    widths  = np.array([w for _,_,_,w,_ in gaps], dtype=float)
    valleys = np.array([v for _,_,_,_,v in gaps], dtype=float)

    width_ratio = float(w_mid / max(1.0, widths.mean()))
    # Loosened but still safe
    deep_enough = v_mid < (valleys.mean() - 0.25*valleys.std())   # was 0.30
    wide_enough = width_ratio >= 1.18                             # was 1.30

    if not (deep_enough or wide_enough):
        return None

    # Build left/right blocks around the mid-gap and score them
    cx = (g0 + g1)//2
    margin = max(1, int(0.02 * W))
    left_up  = up[:, :max(1, cx - margin)]
    right_up = up[:, min(W-1, cx + margin):]

    # Back to original coords for grid search
    left  = cv2.resize(left_up,  None, fx=1/3, fy=1/3, interpolation=cv2.INTER_AREA)
    right = cv2.resize(right_up, None, fx=1/3, fy=1/3, interpolation=cv2.INTER_AREA)

    lb = _best_grid_k(left, 2)
    rb = _best_grid_k(right, 3)

    ls = _score_cells(left, lb)
    rs = _score_cells(right, rb)

    # Slightly looser side checks; right should still dominate a bit
    left_ok  = (ls / max(2, len(lb)))  >= 0.47   # was 0.50
    right_ok = (rs / max(3, len(rb)))  >= 0.47   # was 0.50
    shape_ok = (rs >= ls * 1.005)                # was 1.03

    if not (left_ok and right_ok and shape_ok):
        return None

    # Quality blends width advantage and valley depth
    wr = np.clip((width_ratio - 1.0) / 1.0, 0.0, 1.0)
    vd = np.clip((valleys.mean() - v_mid) / max(1e-6, valleys.mean()), 0.0, 1.0)
    q  = float(0.55*wr + 0.45*vd)

    return {"quality": q, "ls": float(ls), "rs": float(rs)}


# =============== Mid-gap stats for decimal veto =================

def _mid_gap_stats(line_gray: np.ndarray, boxes5: List[tuple[int,int]]) -> Optional[tuple[float, float]]:
    """
    Return (width_ratio, valley_z) for the middle gap (between digits #2 and #3).
    width_ratio: gap width / mean(widths)   (higher = wider than others)
    valley_z:    (valley - mean) / std      (lower/negative = deeper than others)
    """
    if not boxes5 or len(boxes5) != 5:
        return None

    up = cv2.resize(line_gray, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
    up = cv2.equalizeHist(up)
    bw = cv2.threshold(up, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    H, W = bw.shape

    band = bw[int(0.58*H):int(0.94*H), :]
    vp = band.mean(axis=0) / 255.0
    vp_s = _smooth_1d(vp, 11)

    gaps = []
    for i in range(4):
        _, r = boxes5[i]
        l2, _ = boxes5[i+1]
        g0, g1 = int(r*3), int(l2*3)
        if g1 > g0:
            width = g1 - g0
            valley = float(vp_s[g0:g1].mean()) if g1 > g0 else 1.0
            gaps.append((i, width, valley))
    if len(gaps) != 4:
        return None

    # middle gap (index 1 => between 2nd and 3rd digit)
    _, w_mid, v_mid = gaps[1]
    widths  = np.array([w for _, w, _ in gaps], dtype=float)
    valleys = np.array([v for _, _, v in gaps], dtype=float)
    w_ratio = float(w_mid / max(1.0, widths.mean()))
    v_std = float(valleys.std()) if valleys.std() > 1e-9 else 1.0
    v_z = float((v_mid - valleys.mean()) / v_std)   # negative = deeper than mean

    return w_ratio, v_z


# ================= Public API =================

def read_meter(image_bytes: bytes) -> Tuple[str, Dict]:
    """
    1) Crop LCD + digit-line.
    2) Try DECIMAL:
         - real dot (contour) → per-cell (template-only)
         - else: 5-grid NN.NNN hypothesis
         - else: synthetic dot from widest/deepest valley (validated)
    3) Always compute 5-digit integer path and arbitrate by scores.
    """
    bgr = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        return "", {"error": "invalid-image"}

    lcd_bgr = _find_lcd_bgr(bgr)
    gray_lcd = cv2.cvtColor(lcd_bgr, cv2.COLOR_BGR2GRAY)
    line = _crop_digit_line(gray_lcd)

    # ---------- DECIMAL CANDIDATE ----------
    decimal_candidate: Optional[str] = None
    decimal_score: float = -1e9
    dot_found = _find_decimal_dot_x(line)

    # (A) primary: real dot
    if dot_found is not None:
        dot_x, dot_quality = dot_found
        margin = max(1, int(0.02 * line.shape[1]))
        left  = line[:, :max(1, dot_x - margin)]
        right = line[:, min(line.shape[1]-1, dot_x + margin):]

        left_boxes  = _best_grid_k(left, 2)
        right_boxes = _best_grid_k(right, 3)
        left_txt  = re.sub(r"\D+", "", _ocr_cells(left,  left_boxes,  template_only=True) or "")
        right_txt = re.sub(r"\D+", "", _ocr_cells(right, right_boxes, template_only=True) or "")
        left_txt  = (("00"  + left_txt)[-2:]) if left_txt else "00"
        right_txt = (("000" + right_txt)[-3:]) if right_txt else "000"
        cand_real = f"{left_txt}.{right_txt}"

        if re.fullmatch(r"\d{2}\.\d{3}", cand_real):
            # decimal fixup
            L_fixed, R_fixed = _decimal_fixup(left, right, left_boxes, right_boxes, left_txt, right_txt)
            cand_real = f"{L_fixed}.{R_fixed}"

            ls = _score_cells(left, left_boxes)
            rs = _score_cells(right, right_boxes)
            decimal_candidate = cand_real
            decimal_score = ls + rs + 0.5 * dot_quality  # small bonus for good dot

    # (B) secondary: 5-grid NN.NNN if no real-dot candidate
    if decimal_candidate is None:
        cand23, score23, boxes5_tmp = _decimal_from_5grid(line)
        decimal_candidate = cand23
        decimal_score = score23

    # ---------- INTEGER CANDIDATE (ALWAYS) ----------
    boxes5 = _best_grid_k(line, 5)
    digits5 = _ocr_cells(line, boxes5, template_only=False)
    digits5 = re.sub(r"\D+", "", digits5 or "")
    digits5 = digits5[-5:] if len(digits5) >= 5 else (("0"*5) + digits5)[-5:]
    score5  = _score_cells(line, boxes5)

    # (C) synthetic dot from valley — try even if we already have backup decimal
    synthetic_used = False
    syn_best_cand: Optional[str] = None
    syn_best_score: float = -1e9
    if boxes5:
        syn = _synthetic_dot_from_valley(line, boxes5)
        if syn is not None:
            sx, q = syn
            margin = max(1, int(0.02 * line.shape[1]))
            left  = line[:, :max(1, sx - margin)]
            right = line[:, min(line.shape[1]-1, sx + margin):]
            left_boxes  = _best_grid_k(left, 2)
            right_boxes = _best_grid_k(right, 3)
            lt = re.sub(r"\D+", "", _ocr_cells(left,  left_boxes,  template_only=True) or "")
            rt = re.sub(r"\D+", "", _ocr_cells(right, right_boxes, template_only=True) or "")
            lt = (("00"  + lt)[-2:]) if lt else "00"
            rt = (("000" + rt)[-3:]) if rt else "000"
            cand = f"{lt}.{rt}"

            if re.fullmatch(r"\d{2}\.\d{3}", cand):
                # decimal fixup
                L_fixed, R_fixed = _decimal_fixup(left, right, left_boxes, right_boxes, lt, rt)
                cand = f"{L_fixed}.{R_fixed}"

                ls = _score_cells(left, left_boxes)
                rs = _score_cells(right, right_boxes)
                # side validation (avoid false positives on sample #1)
                left_ok  = (ls / max(2, len(left_boxes)))  >= 0.48
                right_ok = (rs / max(3, len(right_boxes))) >= 0.48
                shape_ok = (rs >= ls * 1.02)
                if left_ok and right_ok and shape_ok:
                    syn_best_cand = cand
                    syn_best_score = ls + rs + 0.35 * q

    # If synthetic is stronger (or close), adopt it so dot_used=True later
    if syn_best_cand is not None and syn_best_score >= decimal_score - 0.05:
        decimal_candidate = syn_best_cand
        decimal_score = syn_best_score
        synthetic_used = True

    # (D) promotion of BACKUP decimal if the 2|3 valley is exceptionally strong
    backup_promoted = False
    promo = None
    if (not synthetic_used) and (dot_found is None) and re.fullmatch(r"\d{2}\.\d{3}", decimal_candidate or ""):
        promo = _promote_backup_decimal(line, boxes5)
        if promo is not None:
            # treat as 'dot used' with small margin tuned by quality
            backup_promoted = True
            # small boost to decimal_score so it can edge out ties
            decimal_score = max(decimal_score, (promo["ls"] + promo["rs"]) / 1.0)

    # ---------- DECIMAL VETO FOR STRONG INTEGER SCREENS ----------
    # If there is NO real dot and the decimal came from synthetic/promotion,
    # require an exceptional middle gap; otherwise veto decimal and keep integer.
    avg5 = score5 / max(1, len(boxes5))
    came_from_synth_or_promo = (dot_found is None) and (synthetic_used or backup_promoted)
    if came_from_synth_or_promo:
        stats = _mid_gap_stats(line, boxes5)
        if stats is not None:
            w_ratio, v_z = stats
            # Strong integer screen demands standout mid gap to accept decimal
            if (avg5 >= 0.60) and not (w_ratio >= 1.25 and v_z <= -0.35):
                decimal_candidate = None
                decimal_score = -1e9
                synthetic_used = False
                backup_promoted = False

    # ---------- ARBITRATION ----------
    dot_used = False
    needed_margin: Optional[float] = None
    if re.fullmatch(r"\d{2}\.\d{3}", decimal_candidate or ""):
        if dot_found is not None:
            dot_used = True
            _, q = dot_found
            needed_margin = float(np.clip(0.22 * (1.0 - 0.5*max(0.0, min(1.0, q))), 0.12, 0.30))
        elif synthetic_used:
            dot_used = True
            needed_margin = 0.22
        elif backup_promoted:
            dot_used = True
            # promotion uses small margin that shrinks with quality
            q = float(promo.get("quality", 0.6)) if promo else 0.6
            needed_margin = float(np.clip(0.20 * (1.0 - 0.5*q), 0.14, 0.26))
        else:
            needed_margin = 0.60

        # allow a slightly stronger tie-breaker when a (real/synthetic/promoted) dot is used
        margin_ok = (decimal_score >= score5 + needed_margin) or (dot_used and decimal_score >= score5 - 0.02)
        if margin_ok:
            return decimal_candidate, {
                "tesseract_used": _USE_TESS,
                "lcd_size": list(gray_lcd.shape),
                "ok": True,
                "method": "decimal via dot/valley + per-cell (template-only), scored",
                "scores": {
                    "decimal": round(decimal_score, 3),
                    "integer": round(score5, 3),
                    "needed_margin": round(needed_margin, 3),
                    "dot_used": bool(dot_used),
                },
            }

    # ---------- FALLBACK: integer (protects sample #1) ----------
    return digits5, {
        "tesseract_used": _USE_TESS,
        "lcd_size": list(gray_lcd.shape),
        "ok": True,
        "method": "grid-search IoU → per-cell tesseract+template (5 sanity, last 3→5)",
        "scores": {
            "decimal": round(decimal_score, 3) if isinstance(decimal_score, (int, float)) else None,
            "integer": round(score5, 3),
            "needed_margin": needed_margin,
            "dot_used": bool(dot_used),
        },
    }

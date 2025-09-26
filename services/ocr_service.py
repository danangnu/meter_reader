# services/ocr_service.py
import re
from typing import Tuple, Dict, List
import numpy as np
import cv2

# --------- Optional Tesseract ----------
_USE_TESS = False
try:
    import pytesseract
    # If needed on Windows:
    # pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
    _USE_TESS = True
except Exception:
    _USE_TESS = False


# ============ helpers ============

def _enhance(gray: np.ndarray, fx: float = 4.5) -> np.ndarray:
    g = cv2.resize(gray, None, fx=fx, fy=fx, interpolation=cv2.INTER_CUBIC)
    g = cv2.bilateralFilter(g, 5, 50, 50)
    clahe = cv2.createCLAHE(clipLimit=2.8, tileGridSize=(8, 8))
    g = clahe.apply(g)
    blur = cv2.GaussianBlur(g, (0, 0), 1.2)
    g = cv2.addWeighted(g, 1.6, blur, -0.6, 0)
    return g

def _bin_pair(g: np.ndarray):
    otsu_inv = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    ada_inv  = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY_INV, 31, 7)
    return otsu_inv, ada_inv

def _smooth_1d(x: np.ndarray, k: int = 9) -> np.ndarray:
    k = max(3, k | 1)
    ker = np.ones(k, np.float32) / k
    return np.convolve(x, ker, mode="same")


# ============ LCD crop ============

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
        # fallback window
        lcd = bgr[int(0.10*h):int(0.35*h), int(0.18*w):int(0.82*w)]

    # crop the lower band where digits live (still wide)
    H, W = lcd.shape[:2]
    band = lcd[int(0.24*H):int(0.98*H), int(0.06*W):int(0.94*W)]
    return band if band.size else lcd


# ============ Digit line crop ============

def _crop_digit_line(gray_lcd: np.ndarray) -> np.ndarray:
    g = _enhance(gray_lcd, fx=4.0)
    b1, b2 = _bin_pair(g)
    bw = cv2.bitwise_or(b1, b2)

    # horizontal projection
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

    # trim left/right
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


# ============ WHOLE-LINE / WHOLE-LCD DECIMAL OCR (new) ============

def _ocr_line_decimal(img_gray: np.ndarray) -> str:
    """
    Try to read a decimal value like '04.820' from a grayscale LCD/digit-line image.
    Returns '' if not confidently found.
    """
    if not _USE_TESS:
        return ""

    g0 = cv2.resize(img_gray, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
    g0 = cv2.bilateralFilter(g0, 5, 50, 50)
    g0 = cv2.equalizeHist(g0)

    v_raw = g0
    v_bin = []
    v_bin.append(cv2.threshold(g0, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1])
    v_bin.append(cv2.adaptiveThreshold(g0, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                       cv2.THRESH_BINARY_INV, 31, 7))
    variants = [v_raw] + v_bin + [cv2.bitwise_not(v) for v in v_bin]

    cfgs = [
        "--oem 1 --psm 7  -c tessedit_char_whitelist=0123456789.",
        "--oem 1 --psm 6  -c tessedit_char_whitelist=0123456789.",
        "--oem 1 --psm 11 -c tessedit_char_whitelist=0123456789.",
        "--oem 1 --psm 13 -c tessedit_char_whitelist=0123456789.",
    ]

    strict = re.compile(r"\b\d{1,3}[.]\d{2,3}\b")
    loose  = re.compile(r"\d+[.,]\d+")

    best = ""
    for img in variants:
        for cfg in cfgs:
            try:
                txt = pytesseract.image_to_string(img, config=cfg)
            except Exception:
                continue
            if not txt:
                continue
            t = txt.strip().replace(" ", "")
            t = t.replace("O", "0").replace("o", "0").replace(",", ".").replace("·", ".")
            # find decimal anywhere in the text
            m = strict.findall(t)
            if m:
                cand = max(m, key=len)
                if cand.replace(".", "") != "00000":
                    return cand
            if not best:
                m2 = loose.findall(t)
                if m2:
                    best = m2[0].replace(",", ".")
    return best


# ============ Seven-seg scoring (for 5-digit path) ============

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
        if d == "5":
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


# ============ 5-digit grid search path (kept for sample #1) ============

def _best_grid(line: np.ndarray) -> List[Tuple[int,int]]:
    H, W = line.shape[:2]
    up = cv2.resize(line, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
    UH, UW = up.shape[:2]

    def cell_roi(ix0, ix1):
        return up[:, max(0, ix0):min(UW, ix1)]

    est_w = W / 5.0
    w_min = int(max(9, est_w * 0.70))
    w_max = int(min(W//3, est_w * 1.35))
    gaps  = list(range(1, 6))

    best_score = -1e9
    best_boxes: List[Tuple[int,int]] = []

    widths = np.linspace(w_min, w_max, num=10, dtype=int)
    for cw in widths:
        for gap in gaps:
            span = 5*cw + 4*gap
            if span >= W:
                continue
            base = W - span
            for off in range(max(0, base-10), min(W-span, base+10)+1):
                xs = []
                x = off
                for _ in range(5):
                    xs.append((x, x+cw))
                    x += cw + gap

                score = 0.0
                for (a, b) in xs:
                    a3, b3 = int(a*3), int(b*3)
                    cell = cell_roi(a3, b3)
                    d, s = _classify_template(cell)
                    score += s
                if score > best_score:
                    best_score, best_boxes = score, xs[:]

    if not best_boxes:
        margin = max(1, int(0.01 * W))
        cw = (W - 4*margin) // 5
        best_boxes = [(i*(cw+margin), i*(cw+margin)+cw) for i in range(5)]

    return best_boxes


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

def _ocr_cells(line: np.ndarray, boxes: List[Tuple[int,int]]) -> str:
    up = cv2.resize(line, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
    UH, UW = up.shape[:2]
    out = []

    for idx, (a, b) in enumerate(boxes):
        a3 = max(0, int(a*3)); b3 = min(UW, int(b*3))
        cell = up[:, a3:b3]

        # tighten vertically
        m = (_bin_pair(cell)[0] > 0) | (_bin_pair(cell)[1] > 0)
        hp = m.mean(axis=1)
        rows = np.where(hp > max(0.05, hp.mean() + 0.35*hp.std()))[0]
        if rows.size:
            y0, y1 = rows[0], rows[-1]
            pad = max(2, int(0.02 * m.shape[0]))
            y0 = max(0, y0 - pad); y1 = min(m.shape[0]-1, y1 + pad)
            cell = cell[y0:y1+1, :]

        # Tesseract + Template
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

        # last-digit disambig 3→5 (score + segments)
        if idx == len(boxes) - 1 and ch == "3":
            en = _seg_energies(cmask)
            A,B,C,D,E,F,G = en["A"],en["B"],en["C"],en["D"],en["E"],en["F"],en["G"]
            looks5 = (A>0.18 and D>0.18 and G>0.16 and C>0.14 and F>0.10 and B<0.24 and E<0.24)
            sc = _template_scores(cell); s5, s3 = sc.get("5",0.0), sc.get("3",0.0)
            edge = F > (B + 0.02) and C >= 0.10
            if looks5 or s5 >= 0.92*s3 or edge:
                ch = "5"

        out.append(ch if ch else templ_d)
    return "".join(out)


# ============ Public API ============

def read_meter(image_bytes: bytes) -> Tuple[str, Dict]:
    """
    1) Crop LCD + digit-line.
    2) Try DECIMAL OCR first (digit-line, then whole LCD); if found, return it (e.g., '04.820').
    3) Else: robust 5-digit per-cell pipeline (kept for sample #1).
    """
    bgr = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        return "", {"error": "invalid-image"}

    lcd_bgr = _find_lcd_bgr(bgr)
    gray_lcd = cv2.cvtColor(lcd_bgr, cv2.COLOR_BGR2GRAY)
    line = _crop_digit_line(gray_lcd)

    # ---- DECIMAL FIRST (new) ----
    dec = _ocr_line_decimal(line)
    if not dec:
        dec = _ocr_line_decimal(gray_lcd)  # also try entire LCD
    if dec:
        debug = {
            "tesseract_used": _USE_TESS,
            "lcd_size": list(gray_lcd.shape),
            "ok": True,
            "method": "decimal whole-line tesseract",
        }
        return dec, debug
    # -----------------------------

    # 5-digit fallback (unchanged behavior for sample #1)
    boxes = _best_grid(line)
    digits = _ocr_cells(line, boxes)

    digits = re.sub(r"\D+", "", digits or "")
    digits = digits[-5:] if len(digits) >= 5 else (("0"*5) + digits)[-5:]

    debug = {
        "tesseract_used": _USE_TESS,
        "lcd_size": list(gray_lcd.shape),
        "ok": bool(digits),
        "method": "grid-search IoU → per-cell tesseract+template (5 sanity, last 3→5)",
    }
    return digits, debug
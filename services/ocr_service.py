import io, os, re
from typing import Tuple, Dict, List
import numpy as np
import cv2
from PIL import Image

# ---------- Tesseract setup (Windows-friendly) ----------
_USE_TESS = False
try:
    import pytesseract
    # If Tesseract isn't on PATH, set it explicitly:
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    # (Optional) explicitly point to tessdata if needed:
    # os.environ["TESSDATA_PREFIX"] = r"C:\Program Files\Tesseract-OCR\tessdata"
    _USE_TESS = True
except Exception:
    _USE_TESS = False


# ---------- 1) Locate the LCD area ----------
def _find_lcd_bgr(bgr: np.ndarray) -> np.ndarray:
    """
    Return cropped LCD digits zone (BGR).
    """
    h, w = bgr.shape[:2]

    # Search window where LCD lives
    y1, y2 = int(0.10*h), int(0.35*h)
    x1, x2 = int(0.12*w), int(0.88*w)
    crop = bgr[y1:y2, x1:x2].copy()

    # LCD green/teal tint in HSV
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    lower = np.array([40, 20, 40], dtype=np.uint8)
    upper = np.array([90, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    # Clean and find largest blob
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if cnts:
        x, y, ww, hh = max((cv2.boundingRect(c) for c in cnts), key=lambda b: b[2]*b[3])
        lcd = crop[y:y+hh, x:x+ww]
    else:
        # Fallback rectangle if color mask fails
        lcd = crop[int(0.05*crop.shape[0]):int(0.75*crop.shape[0]),
                   int(0.05*crop.shape[1]):int(0.95*crop.shape[1])]

    # Keep the band where digits are (bottom mid-right), but wide enough for all 4 digits
    H, W = lcd.shape[:2]
    digits_zone = lcd[int(0.35*H):int(0.95*H), int(0.05*W):int(0.96*W)]
    return digits_zone if digits_zone.size else lcd


# ---------- 2) Tesseract OCR (digits-only) ----------
_TESS_CONFIGS = [
    "--oem 1 --psm 7  -c tessedit_char_whitelist=0123456789",
    "--oem 1 --psm 13 -c tessedit_char_whitelist=0123456789",
]

def _tesseract_digits(gray: np.ndarray) -> str:
    """
    Try multiple binarizations + Tesseract configs and return best digit string.
    Prefer a 4-digit sequence; otherwise best 3–6 digits.
    """
    if not _USE_TESS:
        return ""

    versions: List[tuple[str, str]] = []

    def _run(img, tag):
        for cfg in _TESS_CONFIGS:
            try:
                txt = pytesseract.image_to_string(img, config=cfg).strip()
                digits = re.sub(r"\D+", "", txt)
                if digits:
                    versions.append((digits, f"{tag} | {cfg}"))
            except Exception:
                continue

    # Prepare variants
    gray_big = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    _, otsu_inv = cv2.threshold(gray_big, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ada_inv = cv2.adaptiveThreshold(gray_big, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY_INV, 31, 7)

    _run(gray_big, "raw")
    _run(otsu_inv, "otsu_inv")
    _run(ada_inv, "ada_inv")

    if not versions:
        return ""

    # Choose "best": any 3–6 window present? then prefer longer result
    best_digits = max(versions, key=lambda t: (len(re.findall(r"\d{3,6}", t[0])) > 0, len(t[0])))[0]

    # Prefer exact 4-digit sequence (your meter shows 4 digits)
    m = re.findall(r"\d{4}", best_digits)
    if m:
        best_digits = m[-1]
    else:
        m = re.findall(r"\d{3,6}", best_digits)
        if m:
            best_digits = m[-1]

    return best_digits.lstrip("0") or "0"


# ---------- 3) Seven-segment fallback (no Tesseract required) ----------
_SEG_TABLE = {
    (1,1,1,1,1,1,0): '0',
    (0,1,1,0,0,0,0): '1',
    (1,1,0,1,1,0,1): '2',
    (1,1,1,1,0,0,1): '3',
    (0,1,1,0,0,1,1): '4',
    (1,0,1,1,0,1,1): '5',
    (1,0,1,1,1,1,1): '6',
    (1,1,1,0,0,0,0): '7',
    (1,1,1,1,1,1,1): '8',
    (1,1,1,1,0,1,1): '9',
}

def _nearest_digit(pattern: tuple[int, ...]) -> str:
    # Fuzzy match via Hamming distance (tolerate weak segments)
    best, best_d = None, 1e9
    for p, d in _SEG_TABLE.items():
        dist = sum(a != b for a, b in zip(pattern, p))
        if dist < best_d:
            best, best_d = d, dist
    return best or "?"

def _cell_to_pattern(bw: np.ndarray) -> tuple[int, ...]:
    # bw: white segments on black
    img = cv2.resize(bw, (80, 120), interpolation=cv2.INTER_NEAREST)
    A = img[6:24, 20:60]
    B = img[24:60, 56:76]
    C = img[66:102, 56:76]
    D = img[96:114, 20:60]
    E = img[66:102, 4:24]
    F = img[24:60, 4:24]
    G = img[54:72, 20:60]
    segs = [A, B, C, D, E, F, G]

    # FIXED: explicitly index first segment; avoid NameError
    on = [1 if (segs[0] > 128).mean() > 0.20 else 0]
    for s in segs[1:]:
        on.append(1 if (s > 128).mean() > 0.20 else 0)
    return tuple(on)

def _sevenseg_digits(gray: np.ndarray) -> str:
    """
    Split digits zone into 4 equal cells and decode as seven-segment.
    """
    H, W = gray.shape
    # Global local-threshold
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY_INV, 31, 7)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)
    bw = cv2.dilate(bw, np.ones((2, 2), np.uint8), iterations=1)

    margin = int(0.02 * W)
    cell_w = (W - 5 * margin) // 4
    digits = ""

    for i in range(4):
        x0 = margin + i * (cell_w + margin)
        # Refine each cell with Otsu and OR with global mask for robustness
        cell_g = gray[:, x0:x0 + cell_w]
        _, cell_bw = cv2.threshold(cv2.equalizeHist(cell_g), 0, 255,
                                   cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        cell_bw = cv2.bitwise_or(cell_bw, bw[:, x0:x0 + cell_w])
        pat = _cell_to_pattern(cell_bw)
        digits += _nearest_digit(pat)

    return re.sub(r"\D+", "", digits)

def _last_digit_zero_or_nine(gray_digits_zone: np.ndarray, total_digits: int = 4) -> int:
    """
    Disambiguate last digit 0 vs 9 by checking middle segment (G).
    Returns 0 or 9.
    """
    H, W = gray_digits_zone.shape[:2]
    if W < 40 or H < 20:
        return 0

    margin = int(0.02 * W)
    cell_w = (W - (total_digits + 1) * margin) // total_digits
    x0 = margin + (total_digits - 1) * (cell_w + margin)
    last = gray_digits_zone[:, x0:x0 + cell_w]

    last_eq = cv2.equalizeHist(last)
    _, bw = cv2.threshold(last_eq, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    h, w = bw.shape
    mid = bw[int(0.45 * h):int(0.60 * h), int(0.25 * w):int(0.75 * w)]
    on_ratio = (mid > 0).mean()
    return 9 if on_ratio > 0.16 else 0  # a bit lower to catch faint 9s


# ---------- 4) Public API ----------
def read_meter(image_bytes: bytes) -> Tuple[str, Dict]:
    bgr = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        return "", {"error": "invalid-image"}

    lcd = _find_lcd_bgr(bgr)
    gray = cv2.cvtColor(lcd, cv2.COLOR_BGR2GRAY)

    # 1) Tesseract (preferred)
    digits = _tesseract_digits(gray) if _USE_TESS else ""

    # 2) Seven-seg fallback
    if not digits:
        digits = _sevenseg_digits(gray)

    # 3) Cleanup + prefer 4 digits
    digits = re.sub(r"\D+", "", digits)
    if digits:
        m = re.findall(r"\d{4}", digits) or re.findall(r"\d{3,6}", digits)
        if m:
            digits = m[-1]
        digits = digits.lstrip("0") or "0"

    # 4) Last-digit 0/9 fix if we got 4 digits and last is ambiguous
    if len(digits) == 4 and digits[-1] in ("0", "9"):
        try:
            last_val = _last_digit_zero_or_nine(gray, total_digits=4)
            digits = digits[:-1] + str(last_val)
        except Exception:
            pass

    debug = {
        "tesseract_used": _USE_TESS,
        "lcd_size": lcd.shape[:2],  # (H, W)
        "ok": bool(digits),
    }
    return digits, debug

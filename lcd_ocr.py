# lcd_ocr.py
import os, sys, cv2, json, math, shutil
import numpy as np
import pytesseract
from typing import Optional, Tuple

# ====== Tesseract path (Windows) ======
def _setup_tesseract_path() -> Optional[str]:
    fixed = r"C:\Tesseract\tesseract.exe"
    for p in [fixed, os.environ.get("TESSERACT_CMD"), shutil.which("tesseract")]:
        if p and os.path.isfile(p):
            pytesseract.pytesseract.tesseract_cmd = p
            return p
    return None

_tess = _setup_tesseract_path()

# ====== Geometry helpers ======
def order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]   # tl
    rect[2] = pts[np.argmax(s)]   # br
    rect[1] = pts[np.argmin(d)]   # tr
    rect[3] = pts[np.argmax(d)]   # bl
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

    dst = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warp = cv2.warpPerspective(image, M, (W, H))
    if pad > 0:
        warp = cv2.copyMakeBorder(warp, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    return warp

# ====== Color-first LCD detection (green) ======
def find_lcd_quad_color_first(bgr: np.ndarray) -> Optional[np.ndarray]:
    """
    1) Try to find a GREEN rectangular region (typical backlit LCD).
    2) Fallback to generic rectangular contour detection.
    Returns 4x2 float32 or None.
    """
    h, w = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # --- Green mask (tweak if needed) ---
    # Covers yellow-green to cyan-green (broad on purpose):
    lower1 = np.array([35, 40, 50])   # H,S,V
    upper1 = np.array([90, 255, 255])

    mask = cv2.inRange(hsv, lower1, upper1)

    # Clean mask
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, 1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, 2)
    mask = cv2.dilate(mask, kernel, 1)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_score = -1.0
    img_area = float(h * w)

    def _score_rect(rect_pts: np.ndarray, area: float) -> float:
        r = order_points(rect_pts)
        wA = np.linalg.norm(r[1] - r[0])
        wB = np.linalg.norm(r[2] - r[3])
        hA = np.linalg.norm(r[3] - r[0])
        hB = np.linalg.norm(r[2] - r[1])
        width = max(wA, wB)
        height = max(hA, hB)
        if height <= 0:
            return -1
        ar = width / height
        if not (1.4 <= ar <= 12.0):    # allow a bit narrower
            return -1
        # prefer medium-wide shapes
        ar_score = math.exp(-((ar - 4.0) ** 2) / (2 * 2.0 ** 2))
        return (area / img_area) * 1.3 + ar_score * 0.7

    # Try green candidates first
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 0.003 * img_area:
            continue

        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            rect = approx.reshape(4, 2).astype(np.float32)
        else:
            # use minAreaRect -> box
            rot = cv2.minAreaRect(c)
            box = cv2.boxPoints(rot)
            rect = box.astype(np.float32)

        score = _score_rect(rect, area)
        if score > best_score:
            best_score = score
            best = rect

    if best is not None:
        return best

    # --- Fallback: generic rectangular contour from edges ---
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 7, 60, 60)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    v = np.median(gray)
    lo = int(max(0, 0.66 * v))
    hi = int(min(255, 1.33 * v))
    edges = cv2.Canny(gray, lo, hi)
    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3)), 1)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3)), 1)

    cnts2, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_score = -1.0
    for c in cnts2:
        area = cv2.contourArea(c)
        if area < 0.01 * img_area:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) != 4:
            continue
        rect = approx.reshape(4, 2).astype(np.float32)
        hull = cv2.convexHull(rect)
        hull_area = cv2.contourArea(hull)
        if hull_area <= 0:
            continue
        rectangularity = area / hull_area
        if rectangularity < 0.80:
            continue
        score = _score_rect(rect, area)
        if score > best_score:
            best_score = score
            best = rect

    return best

# ====== OCR prep & run ======
def prep_for_ocr(crop: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    g = cv2.bilateralFilter(g, 5, 40, 40)
    g = cv2.resize(g, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)

    # try Otsu + adaptive; pick higher contrast
    _, th_otsu = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adp = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 31, 8)

    def contrast_score(img_bin):
        hist = cv2.calcHist([img_bin], [0], None, [256], [0, 256]).ravel()
        return float(hist[:16].sum() + hist[240:].sum()) / img_bin.size

    th = th_otsu if contrast_score(th_otsu) >= contrast_score(adp) else adp

    # ensure black text on white background
    if np.mean(th) < 128:
        th = cv2.bitwise_not(th)
    return th

def ocr_any(th_img: np.ndarray) -> Tuple[str, str]:
    cfg_general = "--oem 3 --psm 6"
    cfg_digits  = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789.-:/'
    tg = pytesseract.image_to_string(th_img, config=cfg_general)
    td = pytesseract.image_to_string(th_img, config=cfg_digits)

    clean = lambda s: "\n".join([ln.strip() for ln in s.splitlines() if ln.strip()])
    return clean(tg), clean(td)

# ====== Main pipeline ======
def process_image(input_path: str, out_dir: Optional[str] = None) -> dict:
    if out_dir is None:
        out_dir = os.path.dirname(os.path.abspath(input_path)) or "."
    os.makedirs(out_dir, exist_ok=True)

    bgr = cv2.imread(input_path)
    if bgr is None:
        raise FileNotFoundError(f"Cannot read image: {input_path}")

    base = os.path.splitext(os.path.basename(input_path))[0]

    # 1) find LCD quad (green-first)
    quad = find_lcd_quad_color_first(bgr)

    dbg = bgr.copy()
    if quad is not None:
        cv2.polylines(dbg, [quad.astype(int).reshape(-1,1,2)], True, (0,255,0), 2)
        crop = four_point_transform(bgr, quad, pad=4)
        crop_name = base + "_crop.png"
    else:
        # conservative fallback: central crop
        h, w = bgr.shape[:2]
        ch, cw = int(h*0.5), int(w*0.7)
        y0, x0 = max(0,(h-ch)//2), max(0,(w-cw)//2)
        crop = bgr[y0:y0+ch, x0:x0+cw].copy()
        cv2.rectangle(dbg, (x0,y0), (x0+cw, y0+ch), (0,165,255), 2)
        crop_name = base + "_crop_fallback.png"

    dbg_path  = os.path.join(out_dir, base + "_debug.png")
    crop_path = os.path.join(out_dir, crop_name)
    cv2.imwrite(dbg_path, dbg)
    cv2.imwrite(crop_path, crop)

    # 2) OCR
    th = prep_for_ocr(crop)
    general, digits = ocr_any(th)

    annotated = crop.copy()
    overlay = general if general else digits
    if not overlay:
        overlay = "(no text detected)"
    y = 24
    for line in overlay.splitlines():
        cv2.putText(annotated, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2, cv2.LINE_AA)
        y += 26
    ann_path = os.path.join(out_dir, base + "_annotated.png")
    cv2.imwrite(ann_path, annotated)

    return {
        "ok": True,
        "tesseract_cmd": getattr(pytesseract.pytesseract, "tesseract_cmd", None),
        "lcd_found": quad is not None,
        "paths": {
            "debug_with_quad": dbg_path,
            "crop": crop_path,
            "annotated": ann_path,
        },
        "ocr": {
            "general": general,
            "digits_fallback": digits
        }
    }

# ====== CLI ======
def main():
    if len(sys.argv) < 2:
        print("Usage: python lcd_ocr.py <image_path> [out_dir]")
        sys.exit(1)
    image_path = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) >= 3 else None
    try:
        ver = pytesseract.get_tesseract_version()
        print(f"[info] Tesseract {ver} at {getattr(pytesseract.pytesseract,'tesseract_cmd','<PATH>')}")
    except Exception as e:
        print(f"[warn] Could not get Tesseract version: {e}")
    res = process_image(image_path, out_dir)
    print(json.dumps(res, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()

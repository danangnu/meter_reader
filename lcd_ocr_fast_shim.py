# lcd_ocr_fast_shim.py
# Speed-focused wrapper for lcd_ocr_fixed.py
# - Leaves your original file intact
# - Monkey-patches heavy routines to run faster (optional --fast)
# - Adds a robust Windows-safe imread
# - Forwards all original CLI flags to lcd_ocr_fixed.py

import os
import sys
import math
import numpy as np
import cv2
import argparse
import importlib

# -------- Load your original module --------
L = importlib.import_module("lcd_ocr_fixed")

# -------- Super-robust imread with diagnostics --------
def _robust_imread(path_in):
    import io
    from pathlib import Path

    tried = []

    def _note(label, ok):
        tried.append(f"{label}:{'OK' if ok else 'FAIL'}")

    # Normalize
    p = Path(path_in).expanduser()
    path = str(p)
    # 1) cv2.imread
    try:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        _note("cv2.imread", img is not None)
        if img is not None:
            return img
    except Exception:
        _note("cv2.imread", False)

    # 2) np.fromfile -> imdecode (Windows-friendly)
    try:
        data = np.fromfile(path, dtype=np.uint8)
        ok = data.size > 0
        if ok:
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            _note("np.fromfile+imdecode", img is not None)
            if img is not None:
                return img
        else:
            _note("np.fromfile", False)
    except Exception:
        _note("np.fromfile", False)

    # 3) open() -> imdecode
    try:
        with open(path, "rb") as f:
            buf = f.read()
        arr = np.frombuffer(buf, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        _note("open+imdecode", img is not None)
        if img is not None:
            return img
    except Exception:
        _note("open+imdecode", False)

    # 4) PIL fallback
    try:
        from PIL import Image
    except Exception:
        Image = None
    if Image:
        try:
            with Image.open(path) as pil:
                pil = pil.convert("RGB")
                img = np.array(pil)[:, :, ::-1].copy()
                _note("PIL", img is not None)
                if img is not None:
                    return img
        except Exception:
            _note("PIL", False)
    else:
        _note("PIL-import", False)

    # 5) Windows long-path prefix
    if os.name == "nt":
        try:
            long = "\\\\?\\" + path if not path.startswith("\\\\?\\") else path
            img = cv2.imread(long, cv2.IMREAD_COLOR)
            _note("cv2.imread(\\\\?\\)", img is not None)
            if img is not None:
                return img
            data = np.fromfile(long, dtype=np.uint8)
            ok = data.size > 0
            if ok:
                img = cv2.imdecode(data, cv2.IMREAD_COLOR)
                _note("fromfile+imdecode(\\\\?\\)", img is not None)
                if img is not None:
                    return img
            else:
                _note("fromfile(\\\\?\\)", False)
        except Exception:
            _note("longpath", False)

    # Final diagnostic print (once)
    sys.stderr.write(
        "[robust_imread] Could not load image.\n"
        f"  Path given: {path_in}\n"
        f"  Exists?    {os.path.exists(path)}\n"
        f"  Tried:      {', '.join(tried)}\n"
    )
    return None

# Patch the imread inside your loaded module
L.cv2.imread = _robust_imread

# -------- FAST mode toggle --------
FAST_MODE = (os.environ.get("LCD_FAST_MODE", "0") == "1") or ("--fast" in sys.argv)

if FAST_MODE:
    # Tame OpenCV threading a bit to reduce overhead
    try:
        cv2.setNumThreads(2)
    except Exception:
        pass

# -------- Helpers used by FAST routines --------
def _local_norm(img_u8):
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(4,4))
    x = clahe.apply(img_u8)
    return cv2.normalize(x, None, 0, 255, cv2.NORM_MINMAX)

def _ink_fill(roi_bw_u8) -> float:
    return float((roi_bw_u8 < 240).mean())

def _deflate_if_overfilled(roi_bw: np.ndarray, target=0.48, max_iter=3) -> np.ndarray:
    out = roi_bw.copy()
    for _ in range(max_iter):
        fill = _ink_fill(out)
        if fill <= target:
            break
        inv = 255 - out
        inv = cv2.erode(inv, np.ones((3,3), np.uint8), 1)
        inv = cv2.erode(inv, np.ones((1,3), np.uint8), 1)
        t = (inv > 0).astype(np.uint8)*255
        ys, xs = np.where(t > 0)
        if len(xs) > 0 and len(ys) > 0:
            x0, x1 = max(0, xs.min()), min(t.shape[1]-1, xs.max())
            y0, y1 = max(0, ys.min()), min(t.shape[0]-1, ys.max())
            inv = inv[y0:y1+1, x0:x1+1]
        out = 255 - inv
    return out

# -------- FAST autoscale --------
def fast_autoscale_until_legible(crop_bgr: np.ndarray,
                                 min_char_h: int = 48,
                                 max_scale: float = 8.0,
                                 boost_contrast: bool=False):
    scales = [6.5, 7.5]
    best = None
    for scale in scales:
        bw = L.binarize_lcd(crop_bgr, scale=scale, boost_contrast=boost_contrast)
        spans, yband, proj = L.estimate_digit_slots(bw)
        char_h = yband[1] - yband[0] + 1
        if char_h < min_char_h:
            continue
        score = len(spans)
        cand = (score, -scale, bw, spans, yband, proj, scale)
        if best is None or cand > best:
            best = cand
    if best is None:
        bw = L.binarize_lcd(crop_bgr, scale=max_scale, boost_contrast=boost_contrast)
        spans, yband, proj = L.estimate_digit_slots(bw)
        return bw, spans, yband, proj, max_scale
    _, _, bw, spans, yband, proj, scale = best
    return bw, spans, yband, proj, scale

# -------- FAST per-slot OCR --------
def fast_ocr_by_slots(bw: np.ndarray, spans, yband, dump_dir=None, tag=""):
    if not spans:
        return "", [], []
    y0, y1 = yband
    tight = bw[y0:y1+1, :]
    H = tight.shape[0]
    trim = max(1, int(0.04 * H))
    tight = tight[trim:H-trim, :]

    out_digits = []
    slot_rois_for_fix = []
    slot_meta = []

    for idx, (sx, ex) in enumerate(spans, start=1):
        w = ex - sx + 1
        h = tight.shape[0]
        pad_x = max(3, int(0.33 * w))
        pad_y = max(2, int(0.10 * h))
        a = max(0, sx - pad_x)
        b = min(tight.shape[1] - 1, ex + pad_x)
        roi_full = tight[:, a:b+1]
        roi = cv2.copyMakeBorder(roi_full, pad_y, pad_y, 0, 0, cv2.BORDER_REPLICATE)

        # Upsample + normalize
        Ht = 260
        sc = max(1.0, Ht / max(12, roi.shape[0]))
        roi_up = cv2.resize(roi, None, fx=sc, fy=sc, interpolation=cv2.INTER_CUBIC)
        roi_up = _local_norm(roi_up)

        # Two quick variants
        inv = 255 - cv2.GaussianBlur(roi_up, (3,3), 0)
        th_inv = cv2.adaptiveThreshold(inv, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 31, -9)
        t1 = cv2.morphologyEx(th_inv, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), 1)
        t1 = cv2.dilate(t1, np.ones((4,4), np.uint8), 1)
        roi_var1 = 255 - t1

        _, th_otsu = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        t2 = cv2.morphologyEx(th_otsu, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), 1)
        roi_var2 = 255 - t2

        # Pick the more moderate fill and deflate slightly
        rois = [roi_var1, roi_var2]
        fills = [_ink_fill(r) for r in rois]
        pick_idx = int(np.argmin([abs(f - 0.40) for f in fills]))
        roi_pick = _deflate_if_overfilled(rois[pick_idx], target=0.48, max_iter=3)

        # classify via original 7-seg
        d, c = L._classify_7seg_roi(roi_pick)
        if d == "?" or c < 0.22:
            # cheap Tesseract single-char fallback (max 2 tries)
            digit_guess = ""
            tries = 0
            for psm in (10, 13):
                if tries >= 2:
                    break
                s = L.pytesseract.image_to_string(
                    roi_pick, config=f"--oem 3 --psm {psm} -c tessedit_char_whitelist=0123456789 -c classify_bln_numeric_mode=1"
                )
                tries += 1
                s = "".join(ch for ch in s if ch.isdigit())
                if s:
                    digit_guess = s[0]
                    break
            d = digit_guess if digit_guess else d

        out_digits.append(d if (d and d.isdigit()) else "?")
        slot_rois_for_fix.append(roi_pick.copy())

        # meta
        try:
            ls, rs, c_sig, f_sig, b_sig, a_sig, g_eff, d_eff = L._slot_signals(roi_pick)
        except Exception:
            ls=rs=c_sig=f_sig=b_sig=a_sig=g_eff=d_eff=0.0
        _, meta_conf = L._classify_7seg_roi(roi_pick)
        slot_meta.append({
            "index": idx-1,
            "digit": d if d else "?",
            "conf": float(max(0.0, min(1.0, meta_conf))),
            "signals": {
                "left": float(ls), "right": float(rs), "c": float(c_sig), "f": float(f_sig),
                "b": float(b_sig), "a": float(a_sig), "mid": float(g_eff), "bot": float(d_eff),
            }
        })

        if dump_dir:
            os.makedirs(dump_dir, exist_ok=True)
            cv2.imwrite(os.path.join(dump_dir, f"{tag}_slot{idx}.png"), roi_pick)

    # reuse your fixups
    out_fixed = L._sequence_fixups(out_digits, slot_rois_for_fix)
    out_fixed = L._leading_zero_rescue(out_fixed, slot_rois_for_fix, max_slots=2)

    # tail right-bar → '1'
    if out_fixed:
        j = len(out_fixed) - 1
        try:
            ls, rs, c_sig, f_sig, b_sig, a_sig, g_eff, d_eff = L._slot_signals(slot_rois_for_fix[j])
            if out_fixed[j] in {"7","2","3","?","8"}:
                right_bar = rs >= 0.55
                left_weak = ls <= 0.15
                mid_low   = g_eff <= 0.20
                top_low   = a_sig <= 0.40
                if right_bar and left_weak and mid_low and top_low:
                    out_fixed[j] = "1"
        except Exception:
            pass

    return "".join(out_fixed), slot_rois_for_fix, slot_meta

# -------- FAST line-level hints & dots (skip heavy work) --------
def fast_line_level_hints(bgr_crop: np.ndarray, bw: np.ndarray) -> str:
    return ""

def fast_detect_decimal_points_anywhere(bw, spans, yband):
    return []

# -------- Install FAST patches if enabled --------
if FAST_MODE:
    L.autoscale_until_legible = fast_autoscale_until_legible
    L.ocr_by_slots = fast_ocr_by_slots
    L.line_level_hints = fast_line_level_hints
    L.detect_decimal_points_anywhere = fast_detect_decimal_points_anywhere

# -------- CLI delegate with aggressive path resolution --------
def main():
    # Forward original argv but strip our own '--fast'
    forwarded = [a for a in sys.argv[1:] if a != "--fast"]

    # Try to identify the image arg (first non-flag)
    img_idx = None
    for i, a in enumerate(forwarded):
        if not a.startswith("-"):
            img_idx = i
            break

    if img_idx is not None:
        raw = forwarded[img_idx]

        # Candidates to try in order
        from pathlib import Path
        cands = []
        p = Path(raw).expanduser()
        cands.append(p)
        # forward/back slashes variants
        if os.name == "nt":
            cands.append(Path(str(p).replace("/", "\\")))
            cands.append(Path(str(p).replace("\\", "/")))
        # same dir, different common image extensions
        if not p.exists():
            base = p.stem
            parent = p.parent if str(p.parent) not in ("", ".") else Path.cwd()
            for ext in (".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".PNG", ".JPEG"):
                cands.append(parent / f"{base}{ext}")

        # Also try current working dir if name only
        if len(p.parts) == 1:
            for ext in (".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".PNG", ".JPEG"):
                cands.append(Path.cwd() / f"{p.stem}{ext}")

        # Pick the first existing candidate
        chosen = None
        for cp in cands:
            try:
                if cp.exists():
                    chosen = str(cp)
                    break
            except Exception:
                pass

        if chosen:
            forwarded[img_idx] = chosen
        else:
            # Print very explicit guidance and bail early
            sys.stderr.write(
                "[fast_shim] Image file not found by path heuristics.\n"
                f"  Given: {raw}\n"
                "  Tips:\n"
                "   • Verify the file exists on THIS machine/account.\n"
                "   • Try copying the file to the working folder and run:\n"
                "       python lcd_ocr_fast_shim.py power3.jpg --force-slots 6 --boost-contrast --debug-slots --fast\n"
                "   • Or use forward slashes:\n"
                "       C:/Users/Asus/Downloads/power3.jpg\n"
                "   • In PowerShell, check:\n"
                '       Test-Path "C:\\Users\\Asus\\Downloads\\power3.jpg"\n'
            )

    # In fast mode, if user forgot --out, write to current dir
    FAST_MODE = (os.environ.get("LCD_FAST_MODE", "0") == "1") or ("--fast" in sys.argv)
    if FAST_MODE and ("--out" not in forwarded and "-o" not in forwarded):
        forwarded = forwarded + ["--out", "."]

    # Call the original CLI
    sys.argv = [sys.argv[0]] + forwarded
    L.main()
    
if __name__ == "__main__":
    main()
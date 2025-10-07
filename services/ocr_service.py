import os
from typing import Optional

# Import your existing OCR module
import lcd_ocr_fixed as ocr

def run_meter_ocr(
    input_path: str,
    from_crop: bool = False,
    force_slots: Optional[int] = None,
    decimals: Optional[int] = None,
    min_int_digits: Optional[int] = None,
    min_char_height: int = 48,
    boost_contrast: bool = False,
    debug_slots: bool = False,
    dump_slots: bool = False,
    source: str = "auto",  # "auto" | "slot" | "tess"
) -> dict:
    """
    Thin wrapper around lcd_ocr_fixed.process().
    Saves outputs next to the input image by default.
    """

    # Default output directory: next to the image
    out_dir = os.path.dirname(os.path.abspath(input_path)) or "."

    # Call into your pipeline exactly once
    res = ocr.process(
        input_path=input_path,
        out_dir=out_dir,
        from_crop=from_crop,
        prefer_7seg=False,              # kept for compat in your code
        decimals=decimals,
        min_char_h=min_char_height,
        force_slots=force_slots,
        debug_slots=debug_slots,
        dump_slots=dump_slots,
        boost_contrast=boost_contrast,
        min_int_digits=min_int_digits,
        source=source,
    )

    # Optionally: attach a simple, normalized “value” field
    # Prefer tesseract when asked; else stick with selected_source
    value = res.get("ocr", {}).get("best_text") or ""
    res["value"] = value

    return res
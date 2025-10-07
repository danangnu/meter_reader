import os
import shutil
import tempfile
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from services.ocr_service import run_meter_ocr

router = APIRouter()

@router.post("/read-meter")
async def read_meter(
    # Either provide a file OR a local image_path
    file: Optional[UploadFile] = File(default=None),
    image_path: Optional[str] = Form(default=None),

    # Optional processing controls (all Form so they work with multipart)
    from_crop: bool = Form(default=False),
    force_slots: Optional[int] = Form(default=None),
    decimals: Optional[int] = Form(default=None),
    min_int_digits: Optional[int] = Form(default=None),
    min_char_height: int = Form(default=48),
    boost_contrast: bool = Form(default=False),
    debug_slots: bool = Form(default=False),
    dump_slots: bool = Form(default=False),
    source: str = Form(default="auto"),  # "auto" | "slot" | "tess"
):
    """
    POST /ocr/read-meter
    - Send multipart/form-data with either:
      * file=<uploaded image>   OR
      * image_path=<absolute or relative path on server>
    - Optional knobs mirror lcd_ocr_fixed.process().
    """
    # Validate inputs
    if not file and not image_path:
        raise HTTPException(status_code=400, detail="Provide either 'file' or 'image_path'.")

    temp_path = None
    try:
        # Resolve input image path (save upload to a temp file)
        if file:
            suffix = os.path.splitext(file.filename or "upload")[1] or ".png"
            fd, temp_path = tempfile.mkstemp(prefix="meter_", suffix=suffix)
            os.close(fd)
            with open(temp_path, "wb") as out:
                shutil.copyfileobj(file.file, out)
            input_image = temp_path
        else:
            input_image = image_path
            if not os.path.isfile(input_image):
                raise HTTPException(status_code=404, detail=f"image_path not found: {input_image}")

        # Run OCR service
        result = run_meter_ocr(
            input_path=input_image,
            from_crop=from_crop,
            force_slots=force_slots,
            decimals=decimals,
            min_int_digits=min_int_digits,
            min_char_height=min_char_height,
            boost_contrast=boost_contrast,
            debug_slots=debug_slots,
            dump_slots=dump_slots,
            source=source,
        )
        return JSONResponse(content=result)

    finally:
        # Clean up temp upload file
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass
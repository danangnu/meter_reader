from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from services.ocr_service import read_meter

router = APIRouter(prefix="/ocr", tags=["OCR"])

@router.post("/read-meter")
async def read_meter_api(file: UploadFile = File(...), debug: bool = False):
    img_bytes = await file.read()
    digits, info = read_meter(img_bytes)
    if not digits:
        raise HTTPException(status_code=400, detail="No digits detected")
    out = {"meter_reading": digits}
    if debug:
        out["debug"] = info
    return JSONResponse(out)

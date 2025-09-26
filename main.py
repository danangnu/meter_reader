from fastapi import FastAPI
from routers.ocr_router import router as ocr_router

app = FastAPI(title="Meter OCR API", version="1.0")

# Routers
app.include_router(ocr_router)

@app.get("/")
def root():
    return {"ok": True, "service": "Meter OCR API"}

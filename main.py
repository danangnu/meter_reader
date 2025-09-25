from fastapi import FastAPI
from routers import ocr_router

app = FastAPI(title="OCR Meter Reader API")

# include OCR routes
app.include_router(ocr_router.router)

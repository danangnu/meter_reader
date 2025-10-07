from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import ocr_router

def create_app() -> FastAPI:
    app = FastAPI(title="Meter OCR API", version="1.0.0")

    # CORS (tweak as needed)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(ocr_router.router, prefix="/ocr", tags=["ocr"])

    @app.get("/health")
    def health():
        return {"status": "ok"}

    return app

app = create_app()

# Run with: uvicorn app.main:app --host 127.0.0.1 --port 8080 --reload
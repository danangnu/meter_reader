# OCR Meter Reader API

A FastAPI-based service for reading 7-segment LCD meter digits (like prepaid electricity/water meters) using Tesseract OCR and a seven-segment fallback.

---

## 🚀 Run locally

```bash
# create venv
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows

# install dependencies
pip install -r requirements.txt

# run API
uvicorn main:app --reload --port 8080

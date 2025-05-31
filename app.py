import logging
import os
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from fastapi import UploadFile, File
from split_and_transcribe import split_and_transcribe
from pathlib import Path

# импортируем нашу функцию
from portfolio_generator import generate_portfolio

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

app = FastAPI()

@app.get("/portfolio", response_class=FileResponse)
async def serve_portfolio_page():
    return FileResponse(os.path.join(STATIC_DIR, "portfolio_view.html"))


# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

# Отдаём фронтенд
@app.get("/", response_class=FileResponse)
async def serve_index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

class TextRequest(BaseModel):
    text: str

@app.get("/api/health")
async def health():
    return {"status": "alive"}

@app.post("/api/portfolio/text")
async def api_generate(req: TextRequest, request: Request):
    logging.info("Генерация портфолио через модуль")
    try:
        html_content, pdf_bytes = generate_portfolio(req.text)
    except Exception as e:
        logging.exception("Ошибка генерации")
        raise HTTPException(status_code=500, detail=str(e))

    # Сохраняем на диск, чтобы PDF раздавался статически
    html_path = os.path.join(BASE_DIR, "portfolio.html")
    pdf_path = os.path.join(STATIC_DIR, "portfolio.pdf")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    with open(pdf_path, "wb") as f:
        f.write(pdf_bytes)

    # Возвращаем HTML-фрагмент и ссылку на PDF
    pdf_url = request.url_for("static", path="portfolio.pdf")
    return {"html": html_content, "pdf_url": pdf_url}

@app.post("/api/transcribe")
async def api_transcribe(file: UploadFile = File(...)):
    # сохраняем загруженный аудиофайл
    uploads = Path("uploads")
    uploads.mkdir(exist_ok=True)
    input_path = uploads / file.filename
    with open(input_path, "wb") as f:
        f.write(await file.read())

    # вызываем ваш скрипт split_and_transcribe
    # модель можно выбрать "small", "base" и т.д.
    transcript_path = split_and_transcribe(input_path, model="small", language="Russian")

    # читаем полученный файл
    text = transcript_path.read_text(encoding="utf-8")
    return {"text": text}
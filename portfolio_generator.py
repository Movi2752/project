import os
import logging
from typing import Tuple
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from weasyprint import HTML
from bs4 import BeautifulSoup

# Настройки
MODEL_REPO     = "TheBloke/Qwen2-7B-Instruct-GGUF"
MODEL_FILENAME = "Qwen2.5-7B-Instruct.Q4_K_M.gguf"
MODEL_DIR      = "models"
PROMPT_TEMPLATE = "prompt_template.txt"
LAYOUT_TEMPLATE = "layout.html"

def sanitize_html(raw_html: str) -> str:
    """Удаляет markdown-обёртки и избыточные ```."""
    soup = BeautifulSoup(raw_html, "html.parser")
    for tag in soup(text=lambda t: isinstance(t, str) and "```" in t):
        tag.replace_with(tag.replace("```html", "").replace("```", ""))
    return str(soup)

def deduplicate_sections(html: str) -> str:
    """Убирает повторяющиеся <section> по заголовкам."""
    soup = BeautifulSoup(html, "html.parser")
    seen = set()
    for section in soup.find_all("section"):
        h2 = section.find("h2")
        if h2:
            title = h2.text.strip()
            if title in seen:
                section.decompose()
            else:
                seen.add(title)
    return str(soup)

def download_model() -> str:
    """Скачивает модель из Hugging Face, если её нет локально."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, MODEL_FILENAME)
    if not os.path.isfile(path):
        logging.info("Скачивание модели из репозитория %s (%s)...", MODEL_REPO, MODEL_FILENAME)
        try:
            path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILENAME, local_dir=MODEL_DIR)
            logging.info("Модель скачана и сохранена по пути: %s", path)
        except Exception as e:
            logging.exception("Не удалось загрузить модель: %s", e)
            raise
    else:
        logging.info("Модель найдена локально: %s", path)
    return path

def load_text(path: str, desc: str) -> str:
    """Загружает текстовый файл, бросает IOError если нет."""
    if not os.path.isfile(path):
        raise IOError(f"Файл {path} ({desc}) не найден")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def generate_portfolio(presentation: str) -> Tuple[str, bytes]:
    """
    Генерирует HTML и PDF по тексту самопрезентации.
    :param presentation: текст (уже структурированный по разделам [О себе], [Навыки], ...)
    :return: (full_html, pdf_bytes)
    """
    # Простая проверка входных данных
    if not presentation or not presentation.strip():
        logging.error("Текст самопрезентации отсутствует или пуст")
        raise ValueError("Текст самопрезентации отсутствует или пуст")

    # Шаблоны
    prompt_tpl = load_text(PROMPT_TEMPLATE, "шаблон промта")
    layout_tpl = load_text(LAYOUT_TEMPLATE, "шаблон веб-страницы")
    logging.info("Шаблоны загружены (промт и layout)")

    # Готовим текст промта для модели (используем только шаг генерации HTML)
    # Удаляем часть с шагом 1, так как входной текст уже структурирован
    step2_idx = prompt_tpl.find("Шаг 2")
    if step2_idx != -1:
        prompt_tpl = prompt_tpl[step2_idx:]
        newline_idx = prompt_tpl.find("\n")
        if newline_idx != -1:
            prompt_tpl = prompt_tpl[newline_idx+1:]
    # Подставляем текст презентации в шаблон
    prompt = prompt_tpl.replace("{self_presentation}", presentation)
    logging.info("Финальный промт для генерации HTML сформирован (длина %d символов)", len(prompt))

    # Загружаем и инициализируем модель
    model_path = download_model()
    logging.info("Инициализация модели LLaMA из файла %s", model_path)
    try:
        llm = Llama(
            model_path=model_path,
            n_ctx=8192,
            n_threads=os.cpu_count() or 4,
            verbose=False
        )
    except Exception as e:
        logging.exception("Ошибка при загрузке модели: %s", e)
        raise

    # Генерируем HTML-фрагмент
    logging.info("Начало генерации HTML-фрагмента портфолио")
    try:
        resp = llm(prompt, max_tokens=1500, temperature=0.3)
    except Exception as e:
        logging.exception("Ошибка при генерации текста модели: %s", e)
        raise
    generated = resp["choices"][0]["text"].strip()
    logging.info("Модель вернула ответ (%d символов)", len(generated))

    # Проверяем, не вернула ли модель JSON с уточняющими вопросами
    if generated.lstrip().startswith("{") and "<section" not in generated:
        logging.error("Модель вернула запрос на уточнение вместо HTML-фрагмента")
        raise ValueError("Недостаточно данных для генерации портфолио: требуются уточнения")
    if generated.lstrip().startswith("{") and "<section" in generated:
        logging.warning("Обнаружен JSON с вопросами в ответе модели, извлекаем HTML-фрагмент")
        try:
            generated = generated[generated.index("<section"):]
        except Exception as e:
            logging.error("Не удалось извлечь HTML-фрагмент из ответа: %s", e)
            raise ValueError("Некорректный ответ модели (ни HTML, ни запрос уточнения)")

    # Чистим HTML и убираем дубликаты секций
    clean_html = sanitize_html(generated)
    deduped_html = deduplicate_sections(clean_html)
    full_html = layout_tpl.replace("{{content}}", deduped_html)
    logging.info("HTML-фрагмент очищен от лишних конструкций и вставлен в шаблон")

    # Генерируем PDF в память
    pdf_bytes = b""
    try:
        pdf = HTML(string=full_html, base_url=os.getcwd())
        pdf_bytes = pdf.write_pdf()
        logging.info("PDF успешно сгенерирован (%d байт)", len(pdf_bytes))
    except Exception as e:
        logging.error("Ошибка при генерации PDF: %s", e)
        pdf_bytes = b""

    logging.info("Генерация портфолио завершена")
    return full_html, pdf_bytes

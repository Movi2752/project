# split_and_transcribe.py

from pathlib import Path
import sys
import subprocess
from pydub import AudioSegment

CHUNKS_DIR = Path("chunks")
OUTPUT_DIR = Path("output")
SEGMENT_MS = 10 * 60 * 1000  # 10 минут

def transcribe_chunk(chunk_path: Path, model: str, lang: str) -> Path:

    cmd = [
        sys.executable, "-m", "whisper", str(chunk_path),
        "--language", lang,
        "--model", model,
        "--output_format", "txt",
        "--output_dir", str(CHUNKS_DIR),
        "--device", "cpu"
    ]
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as err:
        print(f"[!] Whisper failed on chunk {chunk_path.name}")
        print("=== stdout ===")
        print(err.stdout)
        print("=== stderr ===")
        print(err.stderr)
        raise
    return CHUNKS_DIR / f"{chunk_path.stem}.txt"

def split_and_transcribe(file_path: Path, model: str, language: str = "Russian") -> Path:
    """
    Разбивает file_path на 10‑минутные куски, транскрибирует каждый через Whisper,
    удаляет все промежуточные файлы (mp3 и .txt) и исходник, возвращая итоговый full.txt.
    """
    CHUNKS_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    base = file_path.stem
    full_txt = OUTPUT_DIR / f"{base}_full.txt"
    full_txt.unlink(missing_ok=True)

    try:
        audio = AudioSegment.from_file(str(file_path))
        for i, start in enumerate(range(0, len(audio), SEGMENT_MS), start=1):
            chunk_path = CHUNKS_DIR / f"{base}_part{i}.mp3"
            part_txt   = CHUNKS_DIR / f"{base}_part{i}.txt"

            # создаём mp3‑кусок
            audio[start:start + SEGMENT_MS].export(str(chunk_path), format="mp3")
            print(f"[+] Created chunk: {chunk_path.name}")

            try:
                # транскрибируем
                part_txt = transcribe_chunk(chunk_path, model, language)
                print(f"[+] Transcribed chunk: {part_txt.name}")

                # дописываем в итоговый
                with full_txt.open("a", encoding="utf-8") as out_f, part_txt.open("r", encoding="utf-8") as in_f:
                    out_f.write(in_f.read().strip() + "\n")
            finally:
                # в любом случае удаляем промежуточные файлы
                chunk_path.unlink(missing_ok=True)
                print(f"[–] Deleted chunk file: {chunk_path.name}")
                part_txt.unlink(missing_ok=True)
                print(f"[–] Deleted part txt: {part_txt.name}")

    finally:
        # убираем исходный файл
        file_path.unlink(missing_ok=True)
        print(f"[–] Deleted source file: {file_path.name}")

    print(f"[✓] Full transcript at: {full_txt}")
    return full_txt

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Split audio/video into chunks and transcribe via OpenAI Whisper."
    )
    parser.add_argument("file", type=Path, help="Path to input audio or video file.")
    parser.add_argument("model", choices=["tiny", "base", "small", "medium"],  # убрали 'turbo'
                        help="Which Whisper model to use.")
    parser.add_argument("--lang", default="Russian", help="Language code for transcription.")
    args = parser.parse_args()

    print(split_and_transcribe(args.file, args.model, args.lang))

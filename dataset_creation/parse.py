import re
from pathlib import Path

import pymupdf4llm


def clean_markdown(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"\s+([,.:;!?])", r"\1", text)
    text = re.sub(r"^#+\s+((?:Раздел|Глава|Статья)\s)", r"\1", text, flags=re.MULTILINE)

    return text.strip()


def convert_pdf_to_md(pdf_path: str, output_dir: str = "output"):
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"Файл не найден: {pdf_path}")

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / f"{pdf_path.stem}.md"

    md_text = pymupdf4llm.to_markdown(pdf_path)

    cleaned_text = clean_markdown(md_text)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(cleaned_text)

    return output_file


if __name__ == "__main__":
    convert_pdf_to_md("data/raw/Kodecs.pdf", output_dir="data/processed/")

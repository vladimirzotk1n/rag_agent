import json
import re
from pathlib import Path
from typing import Dict, List, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter


class TKRusParser:
    """
    Парсер Трудового Кодекса РФ с поддержкой чанков для статей.
    Логика: Заголовок захватывает весь текст до следующего заголовка любого уровня.
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 50):
        self.re_section = re.compile(
            r"^Раздел\s+([IVX]+)\.\s*(.*)$", re.IGNORECASE | re.MULTILINE
        )
        self.re_chapter = re.compile(
            r"^Глава\s+(\d+)\.\s*(.*)$", re.IGNORECASE | re.MULTILINE
        )
        self.re_article = re.compile(
            r"^Статья\s+(\d+)\.\s*(.*)$", re.IGNORECASE | re.MULTILINE
        )

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", ";"],
        )

    def _find_all_headers(self, text: str) -> List[Dict]:
        """Находит все заголовки в тексте и сортирует их по позиции"""
        headers = []

        for match in self.re_section.finditer(text):
            headers.append(
                {
                    "type": "section",
                    "pos": match.start(),
                    "id": match.group(1),
                    "title": match.group(2).strip(),
                }
            )

        for match in self.re_chapter.finditer(text):
            headers.append(
                {
                    "type": "chapter",
                    "pos": match.start(),
                    "id": match.group(1),
                    "title": match.group(2).strip(),
                }
            )

        for match in self.re_article.finditer(text):
            headers.append(
                {
                    "type": "article",
                    "pos": match.start(),
                    "id": match.group(1),
                    "title": match.group(2).strip(),
                }
            )

        headers.sort(key=lambda x: x["pos"])
        return headers

    def enrich_content(
        self, article: Dict, chapter: Optional[Dict], section: Dict
    ) -> str:
        """Добавляет контекст раздела и главы в начало статьи"""
        prefix = f"[Раздел {section['roman']}. {section['title']}]"
        if chapter:
            prefix += f" [Глава {chapter['number']}. {chapter['title']}]"
        article_header = f"Статья {article['number']}. {article['title']}"
        enriched = f"{prefix}\n\n{article_header}\n\n{article['content']}"
        return enriched.strip()

    def split_into_chunks(self, text: str) -> List[str]:
        """Разбивает текст на чанки через LangChain Recursive Splitter"""
        return self.splitter.split_text(text)

    def parse(self, text: str) -> Dict:
        """Парсит текст и возвращает структурированный JSON с чанками"""
        headers = self._find_all_headers(text)
        structure = {"source": "TK_RF", "total_articles": 0, "sections": []}

        current_section: Optional[Dict] = None
        current_chapter: Optional[Dict] = None

        for i, header in enumerate(headers):
            next_pos = headers[i + 1]["pos"] if i + 1 < len(headers) else len(text)
            content_start = text.find("\n", header["pos"]) + 1
            content = text[content_start:next_pos].strip()
            content = re.sub(r"\n{3,}", "\n\n", content).strip()

            if header["type"] == "section":
                current_section = {
                    "id": f"section_{header['id']}",
                    "roman": header["id"],
                    "title": header["title"],
                    "chapters": [],
                    "articles": [],
                }
                structure["sections"].append(current_section)
                current_chapter = None

            elif header["type"] == "chapter":
                current_chapter = {
                    "id": f"chapter_{header['id']}",
                    "number": int(header["id"]),
                    "title": header["title"],
                    "articles": [],
                }
                if current_section:
                    current_section["chapters"].append(current_chapter)

            elif header["type"] == "article":
                chunks = self.split_into_chunks(content)  # разбиваем на чанки
                article = {
                    "id": f"article_{header['id']}",
                    "number": int(header["id"]),
                    "title": header["title"],
                    "content": content,
                    "char_count": len(content),
                    "chunks": chunks,  # добавляем чанки
                    "queries": [],
                }
                if current_chapter:
                    current_chapter["articles"].append(article)
                elif current_section:
                    current_section["articles"].append(article)

                structure["total_articles"] += 1

        return structure

    def save_json(self, structure: Dict, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(structure, f, ensure_ascii=False, indent=2)
        print(f"✅ JSON дерево: {path}")

    def save_jsonl(self, structure: Dict, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        count = 0

        with open(path, "w", encoding="utf-8") as f:
            for section in structure["sections"]:
                for chapter in section.get("chapters", []):
                    for article in chapter.get("articles", []):
                        enriched_content = self.enrich_content(
                            article, chapter, section
                        )
                        record = {
                            "id": article["id"],
                            "article_number": article["number"],
                            "article_title": article["title"],
                            "chapter_number": chapter["number"],
                            "chapter_title": chapter["title"],
                            "section_roman": section["roman"],
                            "section_title": section["title"],
                            "content": article["content"],
                            "content_enriched": enriched_content,
                            "chunks": article["chunks"],
                            "metadata": {
                                "section_id": section["id"],
                                "chapter_id": chapter["id"],
                            },
                        }
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
                        count += 1

                for article in section.get("articles", []):
                    enriched_content = self.enrich_content(article, None, section)
                    record = {
                        "id": article["id"],
                        "article_number": article["number"],
                        "article_title": article["title"],
                        "chapter_number": None,
                        "chapter_title": None,
                        "section_roman": section["roman"],
                        "section_title": section["title"],
                        "content": article["content"],
                        "content_enriched": enriched_content,
                        "chunks": article["chunks"],
                        "metadata": {
                            "section_id": section["id"],
                            "chapter_id": None,
                        },
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    count += 1

        print(f"✅ JSONL плоский: {path} ({count} статей)")


if __name__ == "__main__":
    parser = TKRusParser(chunk_size=1000, chunk_overlap=50)

    input_file = "data/processed/Kodecs.md"
    if not Path(input_file).exists():
        print(f"❌ Файл не найден: {input_file}")
        exit()

    with open(input_file, "r", encoding="utf-8") as f:
        text = f.read()

    structure = parser.parse(text)
    parser.save_json(structure, "data/json/tk_rf_tree.json")
    parser.save_jsonl(structure, "data/json/tk_rf_articles.jsonl")
    print("🎉 Парсинг и разбиение на чанки завершены.")

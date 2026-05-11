import json

with open("./data/json/queried_tk_rf_tree.json", "r") as f:
    data = json.load(f)


def emit_article(section, chapter, article, flat_data):
    split = article.get("split")
    queries_per_chunk = article.get("queries", [])

    chapter_header = (
        f"\nГлава {chapter['number']} {chapter['title']}" if chapter else ""
    )
    article_prefix = (
        f"[Раздел {section['roman']} {section['title']} {chapter_header}\n "
        f"Статья {article['number']} {article['title']}]"
    )

    for i, chunk in enumerate(article["chunks"]):
        pairs = queries_per_chunk[i] if i < len(queries_per_chunk) else []
        chunk_data = {
            "section": section["roman"],
            "section_title": section["title"],
            "chapter": chapter["number"] if chapter else None,
            "chapter_title": chapter["title"] if chapter else None,
            "article": article["number"],
            "article_title": article["title"],
            "article_content": f"{article_prefix}\n\n {article['content']}",
            "chunk_content": f"passage: {article_prefix}\n\n {chunk}",
            "split": split,
            "natural_queries": [f"query: {p['natural']}" for p in pairs],
            "rephrased_queries": [f"query: {p['rephrased']}" for p in pairs],
        }
        flat_data.append(chunk_data)


flat_data = []
for section in data["sections"]:
    for chapter in section["chapters"]:
        for article in chapter["articles"]:
            emit_article(section, chapter, article, flat_data)
    for article in section.get("articles", []):
        emit_article(section, None, article, flat_data)

with open("./data/json/queried_tk_rf_flat.json", "w") as f:
    json.dump(flat_data, f, indent=4, ensure_ascii=False)

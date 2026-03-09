import json

with open("./data/json/queried_tk_rf_tree.json", "r") as f:
    data = json.load(f)


flat_data = []
for section in data["sections"]:
    for chapter in section["chapters"]:
        for article in chapter["articles"]:
            for i, chunk in enumerate(article["chunks"]):
                chunk_data = {
                    "section": section["roman"],
                    "section_title": section["title"],
                    "chapter": chapter["number"],
                    "chapter_title": chapter["title"],
                    "article": article["number"],
                    "article_title": article["title"],
                    "article_content": f"[Раздел {section['roman']} {section['title']} \nГлава {chapter['number']} {chapter['title']}\n Статья {article['number']} {article['title']}]\n\n {article['content']}",
                    "chunk_content": f"passage: [Раздел {section['roman']} {section['title']} \nГлава {chapter['number']} {chapter['title']}\n Статья {article['number']} {article['title']}]\n\n {chunk}",
                    "queries": f"query: {article['queries'][i]}",
                }
                flat_data.append(chunk_data)

with open("./data/json/queried_tk_rf_flat.json", "w") as f:
    json.dump(flat_data, f, indent=4, ensure_ascii=False)

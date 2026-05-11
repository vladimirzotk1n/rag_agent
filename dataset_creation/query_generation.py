import asyncio
import json
import os
import random

from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from src.agent.rephraser import rephrase_query
from src.config import settings
from src.utils.prompts import QUERY_GENERATION_PROMPT, QUERY_GENERATION_SYSTEM_PROMPT

load_dotenv()


class NaturalQueries(BaseModel):
    queries: list[str] = Field(
        description="Список бытовых поисковых запросов от пользователей"
    )


client = AsyncOpenAI(
    api_key=os.getenv("API_KEY"), base_url="https://api.aitunnel.ru/v1"
)

TRAIN_RATIO = 0.8
QUERIES_PER_CHUNK = 8
SEED = 42


async def generate_natural_queries(section, chapter, article, chunk):
    chapter_line = f"Глава {chapter['number']}. {chapter['title']}" if chapter else ""
    prompt = QUERY_GENERATION_PROMPT.substitute(
        queries_per_chunk=QUERIES_PER_CHUNK,
        section_roman=section["roman"],
        section_title=section["title"],
        chapter_line=chapter_line,
        article_number=article["number"],
        article_title=article["title"],
        chunk=chunk,
    )
    response = await client.responses.parse(
        model=settings.llm_model,
        input=[
            {"role": "system", "content": QUERY_GENERATION_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        text_format=NaturalQueries,
    )
    return response.output_parsed.queries


def stratified_split(data, train_ratio: float, seed: int):
    """Внутри каждой главы делим статьи 80/20. Главы — страты.
    Статьи без главы (лежащие прямо в разделе) делим в рамках раздела."""
    rng = random.Random(seed)
    splits: dict[tuple[int, int | None, int], str] = {}

    def assign(key_prefix, n: int):
        if n == 0:
            return
        indices = list(range(n))
        rng.shuffle(indices)
        n_train = max(1, int(n * train_ratio))
        train_set = set(indices[:n_train])
        for a_idx in range(n):
            splits[(*key_prefix, a_idx)] = "train" if a_idx in train_set else "test"

    for s_idx, section in enumerate(data["sections"]):
        for c_idx, chapter in enumerate(section["chapters"]):
            assign((s_idx, c_idx), len(chapter["articles"]))
        assign((s_idx, None), len(section.get("articles", [])))
    return splits


async def process_chunk(section, chapter, article, chunk):
    natural = await generate_natural_queries(section, chapter, article, chunk)
    rephrased = await asyncio.gather(*(rephrase_query(q) for q in natural))
    return [{"natural": n, "rephrased": r} for n, r in zip(natural, rephrased)]


async def generate_dataset(data):
    splits = stratified_split(data, TRAIN_RATIO, SEED)

    n_train = sum(1 for v in splits.values() if v == "train")
    n_test = sum(1 for v in splits.values() if v == "test")
    print(f"Split: train={n_train} articles, test={n_test} articles")

    async def process_article(section, chapter, article, key):
        article["split"] = splits[key]
        article["queries"] = []
        for chunk in article["chunks"]:
            pairs = await process_chunk(section, chapter, article, chunk)
            article["queries"].append(pairs)

        chapter_part = f"гл.{chapter['number']} " if chapter else ""
        print(
            f"[{article['split']}] раздел {section['roman']} "
            f"{chapter_part}ст.{article['number']} — "
            f"{sum(len(c) for c in article['queries'])} запросов"
        )

    for s_idx, section in enumerate(data["sections"]):
        for c_idx, chapter in enumerate(section["chapters"]):
            for a_idx, article in enumerate(chapter["articles"]):
                await process_article(section, chapter, article, (s_idx, c_idx, a_idx))
        for a_idx, article in enumerate(section.get("articles", [])):
            await process_article(section, None, article, (s_idx, None, a_idx))

    return data


if __name__ == "__main__":
    with open("./data/json/tk_rf_tree.json", "r") as f:
        data = json.load(f)

    queried_data = asyncio.run(generate_dataset(data))

    with open("./data/json/queried_tk_rf_tree.json", "w") as f:
        json.dump(queried_data, f, indent=4, ensure_ascii=False)

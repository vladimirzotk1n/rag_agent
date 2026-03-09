import json
import os

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

load_dotenv()


class Questions(BaseModel):
    questions: list[str] = Field(description="Список вопросов к фрагменту")


client = OpenAI(api_key=os.getenv("API_KEY"), base_url="https://api.aitunnel.ru/v1")

SYSTEM_PROMPT = (
    "Ты - специалист по Трудовому Кодексу РФ. Отвечай четко на вопросы пользователя."
)


def generate_queries(data):
    for section in data["sections"]:
        for chapter in section["chapters"]:
            for article in chapter["articles"]:
                for chunk in article["chunks"]:
                    prompt = f"""
                        Сгенерируй 4 вопроса к фрагменту статьи ТК РФ.
                        Требования:
                        - Простой русский язык, как в поисковике: вопросы которые могут задать люди.
                        - Без номера статьи в вопросе
                        - Разные типы: факты, ситуации, сравнения
                        - Вопросы должны быть различны

                        ФРАГМЕНТ СТАТЬИ:
                        ##########################################
                        Раздел {section["roman"]}. {section["title"]} 
                        Глава {chapter["number"]}. {chapter["title"]}
                        Статья {article["number"]}. {article["title"]}:
                        {chunk}
                        ##########################################
                    """

                    response = client.responses.parse(
                        model="gpt-4.1-mini",
                        input=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": prompt},
                        ],
                        text_format=Questions,
                    )

                    queries = response.output_parsed.questions
                    article["queries"].append(queries)
    return data


if __name__ == "__main__":
    with open("./data/json/tk_rf_tree.json", "r") as f:
        data = json.load(f)

    queried_data = generate_queries(data)

    with open("./data/json/queried_tk_rf_tree.json", "w") as f:
        json.dump(queried_data, f, indent=4, ensure_ascii=False)

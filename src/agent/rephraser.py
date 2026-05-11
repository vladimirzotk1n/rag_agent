from langchain.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.config import settings
from src.logger_config import logger
from src.utils.prompts import REPHRASER_PROMPT

rephraser_llm = ChatOpenAI(
    model=settings.llm_model,
    api_key=settings.openai_api_key,
    temperature=0,
)


async def rephrase_query(query: str) -> str:
    response = await rephraser_llm.ainvoke(
        [
            SystemMessage(content=REPHRASER_PROMPT),
            HumanMessage(content=query),
        ]
    )
    rephrased = response.content.strip().strip('"').strip()
    logger.info(f"Rephrased query: {query!r} -> {rephrased!r}")
    return rephrased

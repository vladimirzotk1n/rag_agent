from typing import Literal

from langchain.tools import tool

from src.agent.rephraser import rephrase_query
from src.db.get_top import get_top


def make_retrieve_tool(
    mode: Literal["dense", "bm25", "hybrid"] = "hybrid",
    collected: list[dict] | None = None,
):
    @tool
    async def retrieve_context(query: str) -> str:
        """
        Выдает релевантную информацию к запросу (query: str) по Трудовому Кодексу РФ.
        """
        rephrased = await rephrase_query(query)
        hits = await get_top(query=query, sparse_query=rephrased, mode=mode)
        if collected is not None:
            collected.extend(hits)
        return "\n\n".join(hit.get("article_content") for hit in hits)

    return retrieve_context


retrieve_context = make_retrieve_tool()

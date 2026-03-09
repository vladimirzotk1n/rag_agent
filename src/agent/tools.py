from langchain.tools import tool

from src.db.get_top import get_top


# Multihop
@tool
async def retrieve_context(query: str) -> str:
    """
    Выдает релевантную информацию к запросу по Трудовому Кодексу РФ.
    """
    context_metadata = await get_top(query=query)
    context = "\n\n".join(hit.get("article_content") for hit in context_metadata)
    return context

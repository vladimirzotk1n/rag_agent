import asyncio

from langchain.tools import tool

from src.db.get_top import get_top


# Multihop
@tool(response_format="content_and_artifact")
async def retrieve_context(query: str):
    """
    Retrieves information to help response query
    """
    context_metadata = await asyncio.to_thread(get_top, query=query)
    context = "\n\n".join(hit.get("article_content") for hit in context_metadata)
    return context

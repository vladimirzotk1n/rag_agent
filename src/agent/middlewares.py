from langchain.agents.middleware import ModelRequest, dynamic_prompt

from src.agent.context import rephrased_query_ctx
from src.db.get_top import get_top
from src.utils.prompts import SYSTEM_PROMPT


# Single shot
@dynamic_prompt
async def prompt_with_context(request: ModelRequest):
    user_query = request.state["messages"][-1].text
    rephrased = rephrased_query_ctx.get()
    context_metadata = await get_top(query=user_query, sparse_query=rephrased)
    context = "\n\n".join(hit.get("article_content") for hit in context_metadata)
    message = (
        SYSTEM_PROMPT.substitute(context=context)
        + f"\n\nВопрос пользователя:\n {user_query}"
    )
    print(message)
    return message

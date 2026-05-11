from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.redis.aio import AsyncRedisSaver
from langgraph.graph import START, MessagesState, StateGraph
from langsmith import traceable

from src.agent.context import rephrased_query_ctx
from src.agent.middlewares import prompt_with_context
from src.agent.rephraser import rephrase_query
from src.agent.tools import retrieve_context
from src.config import settings
from src.logger_config import logger

load_dotenv()  # для langsmith

llm = ChatOpenAI(model=settings.llm_model, api_key=settings.openai_api_key)


class RAGState(MessagesState):
    rephrased_query: str


class RAGAgent:
    def __init__(self, checkpointer: AsyncRedisSaver, multihop: bool = False):
        self.checkpointer = checkpointer
        self.graph = None
        self.is_initialized = False
        self.multihop = multihop

    async def setup(self):
        if self.is_initialized:
            logger.info("Agent already setuped")
            return

        if self.multihop:
            agent = create_agent(
                model=llm,
                tools=[retrieve_context],
            )
            logger.info("Multihop agent created")
        else:
            agent = create_agent(
                model=llm,
                middleware=[prompt_with_context],
            )
            logger.info("Single-shot agent created")

        async def rephraser_node(state: RAGState):
            user_query = state["messages"][-1].content
            rephrased = await rephrase_query(user_query)
            return {"rephrased_query": rephrased}

        async def rag_agent_node(state: RAGState):
            messages = state["messages"][-50:]
            token = rephrased_query_ctx.set(state.get("rephrased_query"))
            try:
                response = await agent.ainvoke({"messages": messages})
            finally:
                rephrased_query_ctx.reset(token)
            return {"messages": [response["messages"][-1]]}

        builder = StateGraph(RAGState)
        builder.add_node("rephraser_node", rephraser_node)
        builder.add_node("rag_agent_node", rag_agent_node)
        builder.add_edge(START, "rephraser_node")
        builder.add_edge("rephraser_node", "rag_agent_node")
        self.graph = builder.compile(checkpointer=self.checkpointer)
        self.is_initialized = True

    def delete(self):
        if not self.is_initialized:
            return

        self.graph = None
        self.is_initialized = False

    @traceable
    async def __call__(self, user_message: str, thread_id: str = "0"):
        if not self.is_initialized:
            raise RuntimeError("Rag Agent is not initialized!")

        config = {"configurable": {"thread_id": thread_id}}

        input_state = {"messages": [HumanMessage(content=user_message)]}
        async for message, metadata in self.graph.astream(
            input_state, config=config, stream_mode="messages", durability="sync"
        ):
            if (
                message.content
                and isinstance(message, (AIMessage, AIMessageChunk))
                and metadata.get("langgraph_node") == "rag_agent_node"
            ):
                yield message.content

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.redis.aio import AsyncRedisSaver
from langgraph.graph import START, MessagesState, StateGraph
from langsmith import traceable

from src.agent.middlewares import prompt_with_context
from src.agent.tools import retrieve_context
from src.config import settings
from src.logger_config import logger

load_dotenv()  # для langsmith

llm = ChatOpenAI(model=settings.llm_model, api_key=settings.openai_api_key)


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

        async def rag_agent_node(state: MessagesState):
            messages = state["messages"][-50:]
            response = await agent.ainvoke({"messages": messages})
            return {"messages": [response["messages"][-1]]}

        builder = StateGraph(MessagesState)
        builder.add_node("rag_agent_node", rag_agent_node)
        builder.add_edge(START, "rag_agent_node")
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
        async for message, _ in self.graph.astream(
            input_state, config=config, stream_mode="messages", durability="sync"
        ):
            if message.content:
                yield message.content

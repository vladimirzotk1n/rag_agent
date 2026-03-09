import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langgraph.checkpoint.redis.aio import AsyncRedisSaver

from src.agent.agent import RAGAgent
from src.api.routers import router
from src.config import settings
from src.db.delete_db import delete_db
from src.db.init_db import init_db
from src.logger_config import logger

os.environ["REDIS_URL"] = settings.redis_url


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()

    checkpointer = AsyncRedisSaver(settings.redis_url)
    await checkpointer.asetup()

    agent = RAGAgent(checkpointer, multihop=False)
    await agent.setup()
    app.state.agent = agent

    yield

    await delete_db()
    await checkpointer.aclose()
    agent.delete()


logger.info("Initializing FastAPI")

app = FastAPI(title="RAG Ассистент по ТК РФ", lifespan=lifespan)

origins = ["http://localhost:3000"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

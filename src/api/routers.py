import uuid

from fastapi import APIRouter, Body, Depends, Request
from fastapi.responses import StreamingResponse

from src.api.deps import get_agent
from src.logger_config import logger

router = APIRouter()


@router.post("/ask")
async def ask(
    request: Request,
    user_input: str = Body(...),
    agent=Depends(get_agent),
):
    new_cookie = False
    thread_id = request.cookies.get("thread_id")
    if not thread_id:
        thread_id = str(uuid.uuid4())
        new_cookie = True
        logger.info(f"New thread created: {thread_id}")

    async def stream_messages():
        async for msg in agent(user_input, thread_id=thread_id):
            yield msg.encode("utf-8")

    response = StreamingResponse(stream_messages(), media_type="text/event-stream")
    if new_cookie:
        response.set_cookie(
            key="thread_id",
            value=thread_id,
            httponly=True,
            secure=False,
            samesite="lax",
        )
    return response


@router.get("/healthcheck")
def heathcheck():
    return {"status": "ok"}

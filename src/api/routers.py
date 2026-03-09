from fastapi import APIRouter, Body, Depends
from fastapi.responses import StreamingResponse

from src.api.deps import get_agent, get_thread_id

router = APIRouter()


@router.post("/ask")
async def ask(
    user_input: str = Body(...),
    agent=Depends(get_agent),
    thread_id=Depends(get_thread_id),
):
    async def stream_messages():
        async for msg in agent(user_input, thread_id=thread_id):
            yield msg.encode("utf-8")

    return StreamingResponse(stream_messages(), media_type="text/event-stream")


@router.get("/healthcheck")
def heathcheck():
    return {"status": "ok"}

import uuid

from fastapi import Request, Response

from src.logger_config import logger


def get_agent(request: Request):
    return request.app.state.agent


def get_thread_id(request: Request, response: Response):
    thread_id = request.cookies.get("thread_id")
    if not thread_id:
        thread_id = str(uuid.uuid4())
        response.set_cookie(
            key="thread_id",
            value=thread_id,
            httponly=True,
            secure=False,
            samesite="lax",
        )
        logger.info(f"New thread created: {thread_id}")

    return thread_id

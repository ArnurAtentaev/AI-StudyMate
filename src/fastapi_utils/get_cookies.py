import uuid

from fastapi import Request, Response


def cookies_session(request: Request, response: Response):
    session_id = request.cookies.get("session_id")

    if not session_id:
        session_id = str(uuid.uuid4())
        response.set_cookie(key="session_id", value=session_id, httponly=True)
    return session_id

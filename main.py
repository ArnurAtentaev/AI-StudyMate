import logging

import uvicorn

from src.fastapi_utils.get_cookies import cookies_session
from src.graph.states import GlobalState
from src.graph.agent_graph import build_graph

from fastapi import FastAPI, Request, Response


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    force=True,
)

app = FastAPI()


@app.post("/chat")
def chat_endpoint(request: Request, response: Response, question: str):
    session_id = cookies_session(request, response)
    state = GlobalState(session_id=session_id, question=question)
    compiled_graph = build_graph()

    state = compiled_graph.invoke({"session_id": session_id, "question": question})
    return {"answer": state["answer"]}


if __name__ == "__main__":
    uvicorn.run("main:app", reload=True)

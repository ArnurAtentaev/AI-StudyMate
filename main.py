import logging
import tempfile
import shutil
import os

import uvicorn

from src.fastapi_utils.get_cookies import cookies_session
from src.graph.states import GlobalState
from src.graph.agent_graph import build_graph
from src.db.common_action import CommonAction

from fastapi import FastAPI, Request, Response, UploadFile

from src.utils.preprocessing_docs import detect_file_type


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


@app.post("/upload_file")
async def uploading_files(file: UploadFile):
    file_content = await file.read()
    file_type = detect_file_type(file.filename)

    with tempfile.NamedTemporaryFile(delete=False, suffix=file_type) as tmp:
        tmp.write(file_content)
        tmp_path = tmp.name

    db = CommonAction(collection_name="docs_collection")
    db.add_to_chroma(docs=tmp_path, file_type=file_type)

    logging.info("---DOCUMENT UPLOAD---")

    os.remove(tmp_path)
    return {"status": "ok", "filename": file.filename}


@app.get("/check_docs")
def check_docs():
    db = CommonAction(collection_name="docs_collection")
    return {"count": db.vector_db._collection.count()}


if __name__ == "__main__":
    uvicorn.run("main:app", reload=True)

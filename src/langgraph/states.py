from pydantic import BaseModel
from langchain_core.documents import Document


class GlobalState(BaseModel):
    metadata: dict | None = None
    draft_retriever: str | None = None
    question: str | None = None
    context: str | None = None
    answer: str | None = None

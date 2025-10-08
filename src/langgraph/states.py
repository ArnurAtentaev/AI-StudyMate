from pydantic import BaseModel
from langchain_core.documents import Document


class GlobalState(BaseModel):
    docs: list[Document] | None = None
    metadata: dict | None = None
    query: str | None = None
    context: str | None = None
    answer: str | None = None

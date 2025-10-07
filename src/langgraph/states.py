from pydantic import BaseModel
from langchain_core.documents import Document


class GlobalState(BaseModel):
    docs: Document
    metadata: dict
    embeddings: list[float]
    query: list[float]
    results: str
    answer: str

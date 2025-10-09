from pydantic import BaseModel, Field
from langchain_core.documents import Document


class GlobalState(BaseModel):
    result_rag: list[str]
    relevant_documents: list[str]
    question: str
    answer: str


class GradeDocuments(BaseModel):
    """Бинарная оценка релевантности результата RAG."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

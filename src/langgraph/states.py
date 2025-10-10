from typing import Optional, List

from pydantic import BaseModel, Field


class GlobalState(BaseModel):
    """Общие состояния"""

    question: str
    result_rag: Optional[List[str]] = None
    relevant_documents: List[str] = []
    result_google: Optional[str] = None
    answer: Optional[str] = None


class GradeDocuments(BaseModel):
    """Бинарная оценка релевантности результата RAG."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

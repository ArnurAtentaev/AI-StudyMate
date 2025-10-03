import logging
import uuid
from datetime import datetime

from dotenv import load_dotenv

from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders.pdf import UnstructuredPDFLoader
from langchain_chroma import Chroma


logging.basicConfig(
    level=logging.ERROR,
)


EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"


model_kwargs = {"device": "cuda"}
encode_kwargs = {"normalize_embeddings": True}
embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
)


class ActionWithCromaDB:
    def __init__(self, embedding, collection_name: str, persist_directory):
        self.vector_db = Chroma(
            collection_name=collection_name,
            embedding_function=embedding,
            persist_directory=persist_directory,
        )
        self.collections = {}

    def add_docs(self, documents: list, metadatas: list = None):
        if metadatas is None:
            metadatas = [{"date": str(datetime.now())} for _ in documents]

        self.vector_db.add_texts(documents, metadatas=metadatas)

    def query_docs(self, query_text: str, n_results: int = 4):
        results = self.vector_db.similarity_search(query_text, k=n_results)
        return results

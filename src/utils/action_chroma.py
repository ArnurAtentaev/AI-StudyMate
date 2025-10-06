import logging
from datetime import datetime

from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders.pdf import UnstructuredPDFLoader
from langchain_chroma import Chroma


logging.basicConfig(
    level=logging.INFO,
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

        self.vector_db.add_documents(documents, metadatas=metadatas)

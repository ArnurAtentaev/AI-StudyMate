import os
import logging
from dotenv import load_dotenv

from src.utils.preprocessing_docs import PreprocessingDocs, detect_file_type

from langchain.document_loaders import (
    UnstructuredPDFLoader,
    UnstructuredWordDocumentLoader,
)
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from chromadb.config import Settings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.retrievers import MultiQueryRetriever


load_dotenv(".env")
CHROMA_PORT_EXPOSED = os.getenv("CHROMA_PORT_EXPOSED")
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"

model_kwargs = {"device": "cuda"}
encode_kwargs = {"normalize_embeddings": True}


class CommonAction:
    def __init__(self, collection_name: str):
        self.collection_name = collection_name
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )
        self.vector_db = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embedding_model,
            persist_directory="/chroma_data",
        )

    def add_to_chroma(self, docs, file_type):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=600, chunk_overlap=100, separators=["\n\n", "\n", ".", " ", ""]
        )

        if file_type == "pdf":
            preprocessing_docs = PreprocessingDocs(
                splitter=splitter,
                loader=UnstructuredPDFLoader,
            )

        elif file_type == "doc":
            preprocessing_docs = PreprocessingDocs(
                splitter=splitter,
                loader=UnstructuredWordDocumentLoader,
            )

        documents = preprocessing_docs.load_docs(path=docs)
        normalize_documents = preprocessing_docs.text_normalize(documents)
        cleaned_docs = [
            Document(page_content=d.page_content, metadata=d.metadata)
            for d in normalize_documents
        ]

        self.vector_db.add_documents(documents=cleaned_docs)

    def query_docs(self, query_text: str, chain, n_results: int = 3):
        store_retriever = self.vector_db.as_retriever(
            search_kwargs={
                "k": n_results,
            },
            search_type="similarity",
        )
        multi_retriever = MultiQueryRetriever(
            retriever=store_retriever, llm_chain=chain
        )
        results = multi_retriever.invoke(query_text)
        return results

import logging
import os
from datetime import datetime

from src.utils.preprocessing_docs import PreprocessingDocs, detect_file_type

from langchain.document_loaders import (
    UnstructuredPDFLoader,
    UnstructuredWordDocumentLoader,
)
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.retrievers import BM25Retriever, EnsembleRetriever, MultiQueryRetriever


class CommonAction:
    def __init__(self, embedding_model, collection_name: str):
        self.embedding_model = embedding_model
        self.collection_name = collection_name
        self.persist_directory = "./chroma_db"
        self.vector_db = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embedding_model,
            persist_directory=self.persist_directory,
        )

    def add_to_chroma(self, docs):
        ext = detect_file_type(docs)
        if ext == "pdf":
            preprocessing_docs = PreprocessingDocs(
                splitter=RecursiveCharacterTextSplitter,
                loader=UnstructuredPDFLoader,
            )
            documents = preprocessing_docs.load_docs(path=docs)
            normalize_documents = preprocessing_docs.text_normalize(documents)

            cleaned_docs = [
                Document(page_content=d.page_content, metadata=d.metadata)
                for d in normalize_documents
            ]

            self.vector_db.add_documents(documents=cleaned_docs)

        elif ext == "doc":
            preprocessing_docs = PreprocessingDocs(
                splitter=RecursiveCharacterTextSplitter,
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
        docs = self.vector_db.get(include=["documents"])
        docs = [Document(page_content=d) for d in docs["documents"]]

        store_retriever = self.vector_db.as_retriever(
            search_kwargs={
                "k": n_results,
            },
            search_type="similarity",
        )
        bm25_retriever = BM25Retriever.from_documents(documents=docs)
        ensemble_retriever = EnsembleRetriever(
            retrievers=[store_retriever, bm25_retriever], weights=[0.5, 0.5]
        )
        multi_retriever = MultiQueryRetriever(
            retriever=ensemble_retriever, llm_chain=chain
        )
        results = multi_retriever.invoke(query_text)
        return results

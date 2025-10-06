import logging
import os
from datetime import datetime

from src.utils.action_chroma import ActionWithCromaDB
from src.utils.preprocessing_docs import PreprocessingDocs, detect_file_type

from langchain.document_loaders import (
    UnstructuredPDFLoader,
    UnstructuredWordDocumentLoader,
)
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter


class CommonAction:
    def __init__(self, embedding, collection_name: str, persist_directory):
        self.vector_db = Chroma(
            collection_name=collection_name,
            embedding_function=embedding,
            persist_directory=persist_directory,
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

            self.vector_db.add_documents(normalize_documents)

        elif ext == "doc":
            preprocessing_docs = PreprocessingDocs(
                splitter=RecursiveCharacterTextSplitter,
                loader=UnstructuredWordDocumentLoader,
            )
            documents = preprocessing_docs.load_docs(path=docs)
            normalize_documents = preprocessing_docs.text_normalize(documents)

            self.vector_db.add_documents(normalize_documents)

    def query_docs(self, query_text: str, n_results: int = 4):
        retriever = self.vector_db.as_retriever(
            search_kwargs={"k": 3}, search_type="similarity"
        )
        results = retriever.invoke(input=query_text)
        return results

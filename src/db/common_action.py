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


class CommonAction:
    def __init__(self, embedding_model, collection_name: str, persist_directory):
        self.embedding_model = embedding_model
        self.collection_name = collection_name
        self.persist_directory = persist_directory
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

    def query_docs(self, query_text: str, n_results: int = 3):
        retriever = self.vector_db.as_retriever(
            search_kwargs={"k": n_results}, search_type="similarity"
        )
        results = retriever.invoke(input=query_text)
        return results

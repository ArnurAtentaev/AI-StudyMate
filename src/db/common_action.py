import os
from dotenv import load_dotenv

from src.utils.preprocessing_docs import PreprocessingDocs, detect_file_type

from langchain.document_loaders import (
    UnstructuredPDFLoader,
    UnstructuredWordDocumentLoader,
)
from langchain_chroma import Chroma
from chromadb.config import Settings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.retrievers import MultiQueryRetriever


load_dotenv(".env")
CHROMA_PORT_EXPOSED = os.getenv("CHROMA_PORT_EXPOSED")
CHROMA_PORT_SERVICE = os.getenv("CHROMA_PORT_SERVICE")


class CommonAction:
    def __init__(self, embedding_model, collection_name: str):
        self.embedding_model = embedding_model
        self.collection_name = collection_name
        self.persist_directory = "./chroma_db"
        self.vector_db = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embedding_model,
            client_settings=Settings(
                chroma_server_host="localhost",
                chroma_server_http_port=CHROMA_PORT_SERVICE,
            ),
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

from src.db.chroma_connect import ChromaConnect

from dotenv import load_dotenv

from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders.pdf import UnstructuredPDFLoader

EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"


model_kwargs = {"device": "cuda"}
encode_kwargs = {"normalize_embeddings": True}
embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )


class ActionWithCromaDB:
    def __init__(self, engine, embedding):
        self.engine = engine
        self.embedding = embedding
        self.created_collection = None

    def create_collection(self, name, ids: list, documents: list, metadata: list):
        self.collection = self.engine.create_collection(name=name, metadata=metadata)
        return self.collection

    def add_to_collection(self, ids: list, documents: list, metadatas: list):
        if not self.collection:
            raise ValueError(
                "Коллекция ещё не создана. Сначала вызови create_collection()."
            )
        embedding_docunets = self.embedding.
        self.collection.add(
            ids=ids, documents=documents, metadatas=metadatas, embeddings=embedding_model
        )

    def get_collections(self, query_text: list, n_results: int = 5):
        if not self.collection:
            raise ValueError(
                "Коллекция ещё не создана. Сначала вызови create_collection()."
            )
        query_embedding = self.embedding_model.embed_query(query_text)
        results = self.collection.query(
            query_embeddings=query_embedding, n_results=n_results
        )
        return results

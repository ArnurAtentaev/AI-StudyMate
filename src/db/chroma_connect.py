import logging

from src.utils.singleton import SingletonMeta

import chromadb

MODEL_LLM = "meta-llama/Llama-3.1-8B-Instruct"

logging.basicConfig(
    level=logging.INFO,
)


class ChromaConnect(metaclass=SingletonMeta):
    __connection = None

    def get_connection(self):
        if self.__connection is None:
            try:
                self.__connection = chromadb.PersistentClient(
                    path="./chromda_data/chroma.db"
                )
                logging.info("ChromaDB connection successful")
                return self.__connection
            except ConnectionError as c:
                logging.error(f"Ошибка: {c}")
                raise

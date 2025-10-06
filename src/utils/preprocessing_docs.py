import os
import re
from pathlib import Path

from src.abstractions.preprocessing_abc import AbstractPreprocess

from langchain_core.documents import Document


def detect_file_type(file: str) -> str:
    extention = os.path.splitext(file)[1].lower()
    if extention == ".pdf":
        return "pdf"
    elif extention in (".doc", ".docx"):
        return "doc"


class PreprocessingDocs(AbstractPreprocess):
    def __init__(self, splitter, loader):
        self.loader = loader
        self.splitter = splitter

    def load_docs(self, path: Path):
        splitted_docs = self.splitter(
            chunk_size=600, chunk_overlap=100, separators=["\n\n", "\n", ".", " ", ""]
        )
        loader = self.loader(path, mode="single", languages=["ru"])
        docs = loader.load()
        splitted = splitted_docs.split_documents(docs)

        return splitted

    def text_normalize(self, docs) -> list[Document]:
        cleaned = []
        for d in docs:
            text = d.page_content
            text = re.sub(
                r"([A-Za-zА-Яа-яЁё0-9])-\s+([A-Za-zА-Яа-яЁё0-9])", r"\1\2", text
            )
            text = "\n".join(
                line.strip().lower() for line in text.splitlines() if line.strip()
            )

            text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)
            text = re.sub(r"(page|стр\.)\s*\d+", "", text, flags=re.IGNORECASE)

            text = re.sub(r"\n\s*\n", "\n", text)
            text = "\n".join(line.strip() for line in text.splitlines())

            d.page_content = text
            cleaned.append(d)
        return cleaned

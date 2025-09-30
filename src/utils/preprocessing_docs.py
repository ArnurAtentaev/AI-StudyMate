import re

from src.abstractions.preprocessing_abc import AbstractPreprocess

from langchain_community.document_loaders.pdf import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


class PreprocessingPDFDocs(AbstractPreprocess):
    def __init__(self, splitter, loader):
        self.loader = loader
        self.splitter = splitter

    def load_docs(self, path: str):
        splitted_docs = self.splitter(
            chunk_size=1000, chunk_overlap=100, separators=["\n\n", "\n", ".", " ", ""]
        )
        loader = self.loader(path, mode="single", languages=["ru"])
        docs = loader.load()
        splitted = splitted_docs.split_documents(docs)

        return splitted

    def text_normalize(self, docs):
        cleaned = []
        for d in docs:
            text = d.page_content
            text = re.sub(r'([A-Za-zА-Яа-яЁё0-9])-\s+([A-Za-zА-Яа-яЁё0-9])', r'\1\2', text)
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


a = PreprocessingPDFDocs(
    splitter=RecursiveCharacterTextSplitter,
    loader=UnstructuredPDFLoader,
)
splitted_docs = a.load_docs(path="./file.pdf")
print(splitted_docs[0])
print("-----------------------------------------------------------------------------")
doc = a.text_normalize(splitted_docs)
print(doc[0])
print("-----------------------------------------------------------------------------")
print(doc[1])

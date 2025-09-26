from langchain_community.document_loaders.pdf import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pypdf import PdfReader, PdfWriter


def load_documents(path_pdf):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

    loader = UnstructuredPDFLoader(path_pdf)
    splitted_data = loader.load_and_split(text_splitter)
    return splitted_data


def pdf_clipping(reader, writer, path):
    for i in range(69, 103 + 1):
        writer.add_page(reader.pages[i])

    with open(path, "wb") as f:
        writer.write(f)


res = load_documents(path_pdf="./file.pdf")
print(res[0])

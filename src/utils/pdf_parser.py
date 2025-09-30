from pypdf import PdfReader, PdfWriter


def pdf_clipping(reader: PdfReader, writer: PdfWriter, path):
    for i in range(69, 103 + 1):
        writer.add_page(reader.pages[i])

    with open(path, "wb") as f:
        writer.write(f)

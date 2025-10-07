from typing import List
from pathlib import Path

from langchain_core.output_parsers import BaseOutputParser


class LineListOutputParser(BaseOutputParser[List[str]]):
    def parse(self, text: str) -> List[str]:
        lines = text.strip().split("\n")
        return list(filter(None, lines))


def pdf_clipping(reader, writer, path: Path):
    for i in range(69, 103 + 1):
        writer.add_page(reader.pages[i])

    with open(path, "wb") as f:
        writer.write(f)

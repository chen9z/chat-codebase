import logging
import os
import uuid
import warnings
from dataclasses import dataclass
from typing import List

warnings.simplefilter(action='ignore', category=FutureWarning)

from tree_sitter_languages import get_language, get_parser


@dataclass
class Span:
    start: int
    end: int


@dataclass
class Document:
    chunk_id: str = ""
    path: str = ""
    content: str = ""
    score: float = 0.0
    start_line: int = 0
    end_line: int = 0


def get_content(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as file:
        return file.read()


class BaseSplitter:

    def split(self, path: str, text: str) -> list[Document]:
        pass


class DefaultSplitter(BaseSplitter):
    def __init__(self, chunk_size: int = 2000, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, path: str, text: str) -> list[Document]:
        chunks = []
        lines = text.split("\n")
        current_chunk = ""
        current_chunk_size = 0
        start_line = 1
        chunk_id = 1

        for i, line in enumerate(lines, 1):
            line_length = len(line)

            if current_chunk_size + line_length > self.chunk_size:
                # Create a new chunk
                chunks.append(
                    Document(
                        chunk_id=str(uuid.uuid4()),
                        path=path,
                        content=current_chunk.strip(),
                        score=0.0,  # Initialize score as 0
                        start_line=start_line,
                        end_line=i - 1,
                    )
                )

                # Start a new chunk with overlap
                overlap_start = max(0, current_chunk_size - self.chunk_overlap)
                current_chunk = current_chunk[overlap_start:]
                current_chunk_size = len(current_chunk)
                start_line = i - len(current_chunk.split("\n")) + 1
                chunk_id += 1

            current_chunk += line
            current_chunk_size += line_length

        # Add the last chunk if there's any content left
        if current_chunk:
            chunks.append(
                Document(
                    chunk_id=str(uuid.uuid4()),
                    path=path,
                    content=current_chunk.strip(),
                    score=0.0,
                    start_line=start_line,
                    end_line=len(lines),
                )
            )

        return chunks


class CodeSpliter(BaseSplitter):
    def __init__(self, chunk_size: int, chunk_overlap: int, lang: str):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.lang = lang

    def _create_document(self, path, text, start, end) -> Document:
        return Document(chunk_id='', path=path, content=text[start:end], start_line=start, end_line=end)

    def _get_line_number(self, source_code: str, index: int) -> int:
        total_chars = 0
        for line_number, line in enumerate(source_code.splitlines(keepends=True), start=1):
            total_chars += len(line)
            if total_chars > index:
                return line_number - 1
        return line_number

    def _coalesce_chunks(self, chunks: List[Span]) -> List[Span]:
        new_chunks = []
        current_chunk = Span(0, 0)
        for chunk in chunks:
            if chunk.end - current_chunk.start < self.chunk_size:
                current_chunk.end = chunk.end
            else:
                if current_chunk.end > current_chunk.start:
                    new_chunks.append(current_chunk)
                current_chunk = Span(chunk.start, chunk.end)
        if current_chunk.end > current_chunk.start:
            if current_chunk.end - current_chunk.start < self.chunk_size:
                new_chunks[-1].end = current_chunk.end
            else:
                new_chunks.append(current_chunk)
        return new_chunks

    def _connect_chunks(self, chunks: List[Span]):
        for pre, cur in zip(chunks[:-1], chunks[1:]):
            pre.end = cur.start

    def _chunk_node(self, node, text) -> List[Span]:
        span = Span(node.start_byte, node.start_byte)
        chunks = []
        for child in node.children:
            if child.end_byte - child.start_byte > self.chunk_size:
                if span.end > span.start:
                    chunks.append(span)
                span = Span(child.end_byte, child.end_byte)
                if len(child.children) == 0:
                    chunks.append(Span(child.start_byte, child.end_byte))
                else:
                    chunks.extend(self._chunk_node(child, text))
            elif child.end_byte - span.start > self.chunk_size:
                chunks.append(span)
                span = Span(child.start_byte, child.end_byte)
            else:
                span = Span(span.start, child.end_byte)

        if span.end > span.start:
            chunks.append(span)
        return chunks

    def split(self, path: str, text: str) -> list[Document]:
        parser = get_parser(language=self.lang)
        tree = parser.parse(bytes(text, 'utf-8'))

        root_node = tree.root_node
        if not root_node or not root_node or root_node.type == "ERROR":
            logging.error(f"Failed to parse {path}")
            return []

        spans = self._chunk_node(root_node, text)
        self._connect_chunks(spans)
        spans = self._coalesce_chunks(spans)

        documents = []
        for span in spans:
            documents.append(Document(chunk_id='', path=path, content=text[span.start:span.end],
                                      start_line=self._get_line_number(text, span.start),
                                      end_line=self._get_line_number(text, span.end)))
        return documents


def get_splitter_parser(file_path: str) -> BaseSplitter:
    try:
        lang = file_path.split(".")[-1]
        language = get_language(lang)
        if language:
            return CodeSpliter(chunk_size=2000, chunk_overlap=0, lang=lang)
    except Exception as e:
        logging.warn(f"Failed to get language for file: {file_path}")
    return DefaultSplitter(chunk_size=2000, chunk_overlap=100)


def parse(file_path: str) -> list[Document]:
    splitter = get_splitter_parser(file_path)
    text = get_content(file_path)
    return splitter.split(file_path, text)


if __name__ == "__main__":
    java_path = os.path.expanduser(
        "~/workspace/spring-ai/models/spring-ai-qianfan/src/main/java/org/springframework/ai/qianfan/api/QianFanApi.java")
    documents = parse(java_path)
    for doc in documents:
        print(f"===============content length: {len(doc.content)}")
        print(doc.content)

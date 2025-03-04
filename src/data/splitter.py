import logging
import os
import uuid
from dataclasses import dataclass

import tree_sitter_java as ts_java
from tree_sitter import Language, Parser


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


def is_support_file(file_path: str) -> bool:
    return os.path.splitext(file_path)[1] in [
        ".java",
        ".xml",
        ".yml",
        ".yaml",
        ".properties",
        ".sql",
        ".md",
        ".js",
        ".ts",
        ".css",
        ".html",
        ".vue",
        ".py",
        ".go"
    ]


def get_content(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as file:
        return file.read()


class Splitter:
    def __init__(self, chunk_size: int = 2000, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, path: str) -> list[Document]:
        chunks = []

        with open(path, "r", encoding="utf-8") as file:
            lines = file.readlines()

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


class JavaSplitter(Splitter):
    def __init__(self, chunk_size: int = 1500, chunk_overlap: int = 50):
        super().__init__(chunk_size, chunk_overlap)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, path: str) -> list[Document]:
        parser = Parser(Language(ts_java.language()))
        content = get_content(path)
        tree = parser.parse(bytes(content, "utf-8"))

        root_node = tree.root_node
        return self._chunk_node(root_node, 1, path, content, "", 0)

    def _chunk_node(
            self, node, start_line, path, content, current_chunk, last_end_byte
    ):
        if not node or node.type == "ERROR":
            logging.error(f"Failed to parse {path}")
            return []

        chunks = []
        current_start_line = start_line

        def create_document(chunk, s_line, e_line):
            # if len(chunk)>1500:
            #     print(f"path:{path}, content length:{len(chunk)}")
            #     print(f"{content}")
            return Document(
                chunk_id=str(uuid.uuid4()),
                path=path,
                content=chunk,
                start_line=s_line,
                end_line=e_line,
                score=0.0,
            )

        for child in node.children:
            # skip the license header
            if child and child.type == "block_comment" and child.start_byte == 0:
                last_end_byte = child.end_byte
                current_start_line = child.end_point[0] + 1
                continue
            child_text = content[last_end_byte: child.end_byte]
            child_length = len(child_text)
            if len(current_chunk) + child_length > self.chunk_size:
                if current_chunk and (
                        child.type == "block_comment" or child.type == "method_declaration"
                ):
                    chunks.append(
                        create_document(
                            current_chunk, current_start_line, child.start_point[0]
                        )
                    )
                    current_chunk = child_text
                    current_start_line = child.start_point[0] + 1
                else:
                    chunks.extend(
                        self._chunk_node(
                            child,
                            child.start_point[0],
                            path,
                            content,
                            current_chunk,
                            last_end_byte,
                        )
                    )
                    current_chunk = ""
                    current_start_line = child.end_point[0] + 1
            else:
                current_chunk += child_text

            last_end_byte = child.end_byte

        if current_chunk:
            chunks.append(
                create_document(current_chunk, current_start_line, node.end_point[0])
            )

        return chunks


def get_parse(language: str) -> Splitter:
    if language == ".java":
        return JavaSplitter(chunk_size=1500, chunk_overlap=0)
    else:
        return Splitter(chunk_size=2000, chunk_overlap=100)


def parse(file_path: str) -> list[Document]:
    language = os.path.splitext(file_path)[1].lower()
    splitter = get_parse(language)
    return splitter.split(file_path)


if __name__ == "__main__":
    pass

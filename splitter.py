import logging
import os
import uuid
import tree_sitter_java as ts_java
from tree_sitter import Language, Parser, Node, Tree, TreeCursor
from dataclasses import dataclass


@dataclass
class Document:
    chunk_id: str
    path: str
    content: str
    score: float
    start_line: int
    end_line: int


def is_support_file(file_path: str) -> bool:
    return os.path.splitext(file_path)[1] in [".java", ".xml", "yml", ".yaml", ".properties", ".md"]


def get_content(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as file:
        return file.read()


class Splitter:

    def __init__(self, chunk_size: int = 2000, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, path: str) -> list[Document]:
        chunks = []
        with open(path, "r") as file:
            total_token = 0
            cache_line = []
            for index, line in enumerate(file):
                line_tokens = line.split()
                if total_token + len(line_tokens) > self.chunk_size:
                    chunks.append(
                        Document(chunk_id=uuid.uuid4().__str__(), path=path, content="".join(cache_line), score=0.0,
                                 start_line=index - len(cache_line), end_line=index))
                    cache_line = []
                    total_token = 0
                    continue
                cache_line.append(line)
                total_token += len(line_tokens)
            if cache_line:
                chunks.append(
                    Document(chunk_id=uuid.uuid4().__str__(), path=path, content="".join(cache_line), score=0.0,
                             start_line=index - len(cache_line), end_line=index))
        return chunks

    def merge(self, chunks: list[str]) -> list[str]:
        overlapped_chunks = []
        for i in range(len(chunks) - 1):
            chunk = chunks[i] + ' ' + chunks[i + 1][:self.chunk_overlap]
            overlapped_chunks.append(chunk.strip())
        overlapped_chunks.append(chunks[-1])
        return overlapped_chunks


class JavaSplitter(Splitter):
    def __init__(self, chunk_size: int = 1500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, path: str) -> list[Document]:
        parser = Parser(Language(ts_java.language()))
        content = get_content(path)
        tree = parser.parse(bytes(content, "utf-8"))

        root_node = tree.root_node
        return self._chunk_node(root_node, 0, path, content)

    def _chunk_node(self, node, start_line, path, content):
        if not node or node.type == "ERROR":
            logging.error(f"Failed to parse {path}")
            return []

        chunks = []
        current_chunk = ""
        current_start_line = start_line
        current_end_byte = node.start_byte

        def create_document(chunk, s_line, e_line):
            return Document(
                chunk_id=str(uuid.uuid4()),
                path=path,
                content=chunk,
                start_line=s_line,
                end_line=e_line,
                score=0.0
            )

        for child in node.children:
            child_text = content[current_end_byte:child.end_byte]
            child_length = len(child_text)

            if len(current_chunk) + child_length > self.chunk_size:
                if current_chunk:
                    chunks.append(create_document(current_chunk, current_start_line, child.start_point[0]))
                    current_chunk = child_text
                    current_start_line = child.start_point[0]
                elif child_length > self.chunk_size:
                    chunks.extend(self._chunk_node(child, child.start_point[0], path, content))
                    current_chunk = ""
                    current_start_line = child.end_point[0]
                else:
                    current_chunk = child_text
            else:
                current_chunk += child_text

            current_end_byte = child.end_byte

        if current_chunk:
            chunks.append(create_document(current_chunk, current_start_line, node.end_point[0]))

        return chunks


def get_parse(language: str) -> Splitter:
    if language == ".java":
        return JavaSplitter(chunk_size=200, chunk_overlap=20)
    else:
        return Splitter(chunk_size=200, chunk_overlap=100)


def parse(file_path: str) -> list[Document]:
    language = os.path.splitext(file_path)[1].lower()
    splitter = get_parse(language)
    return splitter.split(file_path)


if __name__ == '__main__':
    # path = "~/workspace/chat-codebase/001.txt"
    # splitter = Splitter(chunk_size=2, chunk_overlap=1)
    # documents = splitter.split(os.path.expanduser(path))
    # for d in documents:
    #     print(d)

    path = os.path.expanduser(
        "~/workspace/spring-ai/spring-ai-core/src/main/java/org/springframework/ai/chat/memory/InMemoryChatMemory.java")
    parser = get_parse(os.path.splitext(path)[1])
    results = parser.split(path)
    for chunk in results:
        print("=================")
        print(chunk.content)
        # pass

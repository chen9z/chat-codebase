import logging
import os
import uuid
import tree_sitter_java as ts_java
from tree_sitter import Language, Parser, Node
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
        paser = Parser(Language(ts_java.language()))
        tree = paser.parse(bytes(get_content(path), "utf-8"))

        root_node = tree.root_node
        chunks = []
        if not root_node or tree.root_node.type == "ERROR":
            logging.error(f"Failed to parse {path}")
            return chunks
        chunks = self.traverse_node(root_node)

        start = root_node.start_byte
        end = root_node.end_byte

        print(str(tree))
        print(tree.root_node)
        print(tree.root_node.text)
        print(tree.root_node.start_byte)
        print(tree.root_node.end_byte)
        return chunks

    def traverse_node(self, node) -> list[Document]:
        chunks = []
        if not node or node.type == "ERROR":
            logging.error(f"Failed to parse {path}")
            return chunks

        return self.chunk_node(node, chunks)

    def chunk_node(self, node, chunks: list[Document]) -> list[Document]:
        start = node.start_byte
        end = node.end_byte
        if end - start < self.chunk_size:
            chunks.append(
                Document(chunk_id=uuid.uuid4().__str__(), path=path, content=node.text, score=0.0, start_line=0,
                         end_line=node.text.splitlines() + 1))
            return chunks
        if not node or node.type == "ERROR":
            logging.error(f"Failed to parse {path}")
            return chunks
        if node.type == "block_comment" and node.start_byte == 0:
            print(f"license_block_comment: {node.text}")
        elif node.type == "package_declaration":
            print(f"package_declaration: {node.text}")
        elif node.type == "block_comment":
            print(f"block_comment: {node.text}")
        elif node.type == "class_declaration":
            print(f"class_declaration {node.text}")
        if node.child_count == 0:
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
    parser.split(path)

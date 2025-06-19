import logging
import os
import sys
import uuid
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List

# Third party imports
from tree_sitter_languages import get_language, get_parser

# Local imports
from src.config import settings
from src.config.parser_config import (
    is_semantic_boundary,
    is_comment_node,
    is_documentation_comment
)

warnings.simplefilter(action='ignore', category=FutureWarning)


# Parser cache to avoid recreating parsers
_parser_cache = {}


def get_cached_parser(language: str):
    """Get a cached parser for the given language."""
    if language not in _parser_cache:
        try:
            # Try the new API first
            lang = get_language(language)
            _parser_cache[language] = lang.parser()
        except Exception as e1:
            try:
                # Fallback to old API
                _parser_cache[language] = get_parser(language)
            except Exception as e2:
                logging.warning(f"Failed to create parser for language {language}: {e1}, {e2}")
                return None
    return _parser_cache[language]


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
                        chunk_id="",
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
        return Document(chunk_id="", path=path, content=text[start:end], start_line=start, end_line=end)

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
            new_chunks.append(current_chunk)
        return new_chunks

    def _connect_chunks(self, chunks: List[Span]):
        for pre, cur in zip(chunks[:-1], chunks[1:]):
            pre.end = cur.start

    def _is_semantic_boundary(self, node) -> bool:
        """Check if a node represents a semantic boundary (class, function, etc.)."""
        return is_semantic_boundary(node.type, self.lang)

    def _find_associated_comments(self, semantic_node, all_nodes, text) -> List:
        """Find comments that should be associated with a semantic node (like Javadoc for methods)."""
        associated_comments = []
        semantic_start = semantic_node.start_byte

        # Look for comments that appear immediately before the semantic node
        for node in all_nodes:
            if is_comment_node(node.type, self.lang):
                # Check if this comment is immediately before the semantic node
                # Allow for some whitespace between comment and semantic node
                comment_end = node.end_byte

                # Get text between comment and semantic node
                between_text = text[comment_end:semantic_start].strip()

                # If there's only whitespace between comment and semantic node,
                # and the comment is a documentation comment, associate them
                if (not between_text or between_text.isspace()) and \
                   is_documentation_comment(node.type, self.lang):
                    associated_comments.append(node)
                elif comment_end < semantic_start and \
                     semantic_start - comment_end < 200:  # Within 200 chars
                    # For line comments immediately before semantic nodes
                    lines_between = text[comment_end:semantic_start].count('\n')
                    if lines_between <= 2:  # At most 2 newlines between
                        associated_comments.append(node)

        return associated_comments

    def _get_extended_span_with_comments(self, semantic_node, all_nodes, text) -> Span:
        """Get a span that includes the semantic node and its associated comments."""
        associated_comments = self._find_associated_comments(semantic_node, all_nodes, text)

        if not associated_comments:
            return Span(semantic_node.start_byte, semantic_node.end_byte)

        # Find the earliest comment start and latest semantic node end
        min_start = min(comment.start_byte for comment in associated_comments)
        min_start = min(min_start, semantic_node.start_byte)
        max_end = max(semantic_node.end_byte,
                     max(comment.end_byte for comment in associated_comments))

        return Span(min_start, max_end)

    def _collect_all_nodes(self, node) -> List:
        """Recursively collect all nodes in the tree."""
        nodes = [node]
        for child in node.children:
            nodes.extend(self._collect_all_nodes(child))
        return nodes

    def _chunk_node(self, node, text) -> List[Span]:
        span = Span(node.start_byte, node.start_byte)
        chunks = []

        # Collect all nodes for comment association
        all_nodes = self._collect_all_nodes(node)

        for child in node.children:
            child_size = child.end_byte - child.start_byte

            # If child is a semantic boundary and reasonably sized, treat as separate chunk
            if self._is_semantic_boundary(child) and child_size <= self.chunk_size * 1.5:
                if span.end > span.start:
                    chunks.append(span)

                # Get extended span that includes associated comments
                extended_span = self._get_extended_span_with_comments(child, all_nodes, text)
                chunks.append(extended_span)
                span = Span(extended_span.end, extended_span.end)

            elif child_size > self.chunk_size:
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
        try:
            parser = get_cached_parser(self.lang)
            if parser is None:
                logging.warning(f"No parser available for language: {self.lang}, falling back to default splitter")
                fallback_splitter = DefaultSplitter(self.chunk_size, self.chunk_overlap)
                return fallback_splitter.split(path, text)

            tree = parser.parse(bytes(text, 'utf-8'))
            root_node = tree.root_node

            if not root_node or root_node.type == "ERROR":
                logging.warning(f"Parse error for {path}, falling back to default splitter")
                fallback_splitter = DefaultSplitter(self.chunk_size, self.chunk_overlap)
                return fallback_splitter.split(path, text)

            spans = self._chunk_node(root_node, text)
            self._connect_chunks(spans)
            spans = self._coalesce_chunks(spans)

            documents = []
            for span in spans:
                documents.append(Document(chunk_id='', path=path, content=text[span.start:span.end],
                                          start_line=self._get_line_number(text, span.start),
                                          end_line=self._get_line_number(text, span.end)))
            return documents

        except Exception as e:
            logging.error(f"Unexpected error parsing {path}: {e}, falling back to default splitter")
            fallback_splitter = DefaultSplitter(self.chunk_size, self.chunk_overlap)
            return fallback_splitter.split(path, text)


def get_splitter_parser(file_path: str) -> BaseSplitter:
    try:
        suffix = Path(file_path).suffix
        lang = settings.ext_to_lang.get(suffix)
        if lang:
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

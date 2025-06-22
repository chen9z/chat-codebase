import logging
import os
import sys
import uuid
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
from enum import Enum

# Third party imports
try:
    import tree_sitter
    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False
    logging.warning("tree-sitter not available, will use default splitter only")

# Local imports
from src.config import settings
from src.config.parser_config import (
    is_semantic_boundary,
    is_comment_node,
    is_documentation_comment
)

if TREE_SITTER_AVAILABLE:
    from src.config.tree_sitter_config import get_parser, get_language
else:
    def get_parser(language: str):
        return None
    def get_language(language: str):
        return None

warnings.simplefilter(action='ignore', category=FutureWarning)


def get_parser_for_language(language: str):
    """Get a parser for the given language."""
    try:
        # Try the new API first
        lang = get_language(language)
        if lang is not None:
            return lang.parser()
        else:
            # Fallback to old API
            parser = get_parser(language)
            if parser is not None:
                return parser
            else:
                logging.warning(f"No parser available for language {language}")
                return None
    except Exception as e:
        logging.warning(f"Failed to create parser for language {language}: {e}")
        return None


class SemanticType(Enum):
    """语义类型枚举，包含类型名称和优先级"""
    # 高优先级 - 独立性强的语义单元
    CLASS = ("class", 10)
    INTERFACE = ("interface", 10)
    ENUM = ("enum", 9)
    FUNCTION = ("function", 8)
    METHOD = ("method", 8)

    # 中优先级 - 相关的辅助内容
    CONSTRUCTOR = ("constructor", 6)
    PROPERTY = ("property", 5)
    COMMENT_BLOCK = ("comment_block", 4)

    # 低优先级 - 可合并的内容
    IMPORT = ("import", 2)
    VARIABLE = ("variable", 1)
    OTHER = ("other", 0)

    def __init__(self, type_name, priority):
        self.type_name = type_name
        self.priority = priority


@dataclass
class Span:
    start: int
    end: int


@dataclass
class SemanticChunk:
    """语义块，包含语义信息的代码片段"""
    span: Span
    semantic_type: SemanticType
    node_type: str  # 原始 AST 节点类型
    name: str = ""  # 提取的名称（类名、函数名等）
    associated_comments: List = field(default_factory=list)
    can_merge: bool = True
    size: int = 0


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

        # Handle empty or whitespace-only text
        if not text or not text.strip():
            return chunks

        lines = text.split("\n")

        if not lines:
            return chunks

        i = 0
        while i < len(lines):
            current_lines = []
            current_size = 0
            start_line_num = i + 1  # 1-based line number

            # Build current chunk
            while i < len(lines):
                line = lines[i]
                line_with_newline = line + "\n" if i < len(lines) - 1 else line
                line_length = len(line_with_newline)

                # Check if adding this line would exceed chunk size
                if current_size + line_length > self.chunk_size and current_lines:
                    break

                current_lines.append(line)
                current_size += line_length
                i += 1

            # Create document for current chunk
            if current_lines:
                chunk_content = "\n".join(current_lines)
                chunks.append(
                    Document(
                        chunk_id=str(uuid.uuid4()),
                        path=path,
                        content=chunk_content,
                        score=0.0,
                        start_line=start_line_num,
                        end_line=start_line_num + len(current_lines) - 1,
                    )
                )

                # Calculate overlap for next chunk - simplified logic
                if self.chunk_overlap > 0 and i < len(lines):
                    # Calculate how many lines to overlap based on character count
                    overlap_size = 0
                    overlap_lines = 0

                    # Start from the end and work backwards
                    for j in range(len(current_lines) - 1, -1, -1):
                        line = current_lines[j]
                        line_size = len(line) + 1  # +1 for newline

                        if overlap_size + line_size <= self.chunk_overlap:
                            overlap_size += line_size
                            overlap_lines += 1
                        else:
                            break

                    # Move back to create overlap, but ensure we don't go backwards
                    if overlap_lines > 0 and overlap_lines < len(current_lines):
                        i -= overlap_lines

        return chunks


class CodeSplitter(BaseSplitter):
    def __init__(self, chunk_size: int, chunk_overlap: int, lang: str):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.lang = lang

    def _get_semantic_type(self, node_type: str) -> SemanticType:
        """根据AST节点类型确定语义类型"""
        type_mapping = {
            # Python
            "class_definition": SemanticType.CLASS,
            "function_definition": SemanticType.FUNCTION,
            "async_function_definition": SemanticType.FUNCTION,
            "decorated_definition": SemanticType.FUNCTION,

            # Java
            "class_declaration": SemanticType.CLASS,
            "interface_declaration": SemanticType.INTERFACE,
            "method_declaration": SemanticType.METHOD,
            "constructor_declaration": SemanticType.CONSTRUCTOR,
            "enum_declaration": SemanticType.ENUM,
            "annotation_type_declaration": SemanticType.INTERFACE,

            # JavaScript/TypeScript
            "class_declaration": SemanticType.CLASS,
            "interface_declaration": SemanticType.INTERFACE,
            "function_declaration": SemanticType.FUNCTION,
            "method_definition": SemanticType.METHOD,
            "arrow_function": SemanticType.FUNCTION,
            "function_expression": SemanticType.FUNCTION,
            "generator_function_declaration": SemanticType.FUNCTION,
            "type_alias_declaration": SemanticType.INTERFACE,
            "enum_declaration": SemanticType.ENUM,

            # Go
            "function_declaration": SemanticType.FUNCTION,
            "method_declaration": SemanticType.METHOD,
            "type_declaration": SemanticType.CLASS,
            "interface_type": SemanticType.INTERFACE,
            "struct_type": SemanticType.CLASS,

            # Rust
            "function_item": SemanticType.FUNCTION,
            "impl_item": SemanticType.CLASS,
            "struct_item": SemanticType.CLASS,
            "enum_item": SemanticType.ENUM,
            "trait_item": SemanticType.INTERFACE,
            "mod_item": SemanticType.CLASS,

            # C/C++
            "function_definition": SemanticType.FUNCTION,
            "class_specifier": SemanticType.CLASS,
            "struct_specifier": SemanticType.CLASS,
            "namespace_definition": SemanticType.CLASS,
            "template_declaration": SemanticType.FUNCTION,
            "union_specifier": SemanticType.CLASS,
            "enum_specifier": SemanticType.ENUM,
        }

        return type_mapping.get(node_type, SemanticType.OTHER)

    def _extract_name(self, node, text: str) -> str:
        """提取节点的名称（类名、函数名等）"""
        try:
            # 尝试找到名称节点
            for child in node.children:
                if child.type == "identifier":
                    return text[child.start_byte:child.end_byte]

            # 如果没有找到identifier，返回节点类型
            return node.type
        except:
            return "unknown"

    def _get_line_number(self, source_code: str, index: int) -> int:
        """Get the line number for a given character index in the source code."""
        if index < 0:
            return 1

        # Handle edge case where index is at or beyond the end of the text
        if index >= len(source_code):
            return len(source_code.splitlines()) or 1

        total_chars = 0
        for line_number, line in enumerate(source_code.splitlines(keepends=True), start=1):
            if total_chars + len(line) > index:
                return line_number
            total_chars += len(line)

        # Fallback - should not reach here normally
        return len(source_code.splitlines()) or 1

    def _collect_comments_once(self, root_node) -> List:
        """一次性收集所有注释节点，按位置排序"""
        comments = []

        def traverse(node):
            if is_comment_node(node.type, self.lang):
                comments.append(node)
            for child in node.children:
                traverse(child)

        traverse(root_node)
        # 按位置排序，便于后续查找
        return sorted(comments, key=lambda n: n.start_byte)

    def _find_comments_for_node(self, semantic_node, all_comments: List, text: str) -> List:
        """为特定语义节点查找关联的注释"""
        associated_comments = []
        semantic_start = semantic_node.start_byte

        # 只检查语义节点之前的注释
        for comment in all_comments:
            if comment.end_byte > semantic_start:
                break  # 注释已排序，后面的都在语义节点之后

            # 检查距离和关联性
            between_text = text[comment.end_byte:semantic_start].strip()

            if not between_text and is_documentation_comment(comment.type, self.lang):
                associated_comments.append(comment)
            elif semantic_start - comment.end_byte < 200:  # Within 200 chars
                lines_between = text[comment.end_byte:semantic_start].count('\n')
                if lines_between <= 2:  # At most 2 newlines between
                    associated_comments.append(comment)

        return associated_comments

    def _is_semantic_boundary(self, node) -> bool:
        """Check if a node represents a semantic boundary (class, function, etc.)."""
        return is_semantic_boundary(node.type, self.lang)

    def _parse_semantic_chunks(self, root_node, text: str) -> List[SemanticChunk]:
        """第一阶段：解析所有语义块"""
        chunks = []
        comments = self._collect_comments_once(root_node)  # 一次性收集注释

        def traverse(node):
            if self._is_semantic_boundary(node):
                semantic_type = self._get_semantic_type(node.type)
                chunk_size = node.end_byte - node.start_byte

                chunk = SemanticChunk(
                    span=Span(node.start_byte, node.end_byte),
                    semantic_type=semantic_type,
                    node_type=node.type,
                    name=self._extract_name(node, text),
                    size=chunk_size
                )

                # 关联注释（现在只需要为这个特定节点查找）
                chunk.associated_comments = self._find_comments_for_node(
                    node, comments, text
                )

                chunks.append(chunk)

            # 继续遍历子节点
            for child in node.children:
                traverse(child)

        traverse(root_node)
        return sorted(chunks, key=lambda c: c.span.start)

    def _create_span_with_comments(self, chunk: SemanticChunk) -> Span:
        """创建包含注释的span"""
        if not chunk.associated_comments:
            return chunk.span

        # 找到最早的注释开始和最晚的结束
        min_start = min(comment.start_byte for comment in chunk.associated_comments)
        min_start = min(min_start, chunk.span.start)
        max_end = max(chunk.span.end,
                     max(comment.end_byte for comment in chunk.associated_comments))

        return Span(min_start, max_end)

    def _merge_group(self, group: List[SemanticChunk]) -> List[Span]:
        """合并一组语义块"""
        if not group:
            return []

        # 按优先级排序，高优先级的放在前面
        group.sort(key=lambda c: c.semantic_type.priority, reverse=True)

        # 计算合并后的范围
        start = min(c.span.start for c in group)
        end = max(c.span.end for c in group)

        # 包含所有相关注释
        all_comments = []
        for chunk in group:
            if chunk.associated_comments:
                all_comments.extend(chunk.associated_comments)

        if all_comments:
            comment_start = min(c.start_byte for c in all_comments)
            start = min(start, comment_start)

        return [Span(start, end)]

    def _merge_by_priority(self, semantic_chunks: List[SemanticChunk], text: str) -> List[Span]:
        """第二阶段：按优先级和大小智能合并"""
        if not semantic_chunks:
            return []

        merged_spans = []
        current_group = []
        current_size = 0

        for chunk in semantic_chunks:
            chunk_size = chunk.size

            # 高优先级的大块独立成chunk
            if (chunk.semantic_type.priority >= 8 and
                chunk_size > self.chunk_size * 0.3):

                # 先处理当前组
                if current_group:
                    merged_spans.extend(self._merge_group(current_group))
                    current_group = []
                    current_size = 0

                # 高优先级块独立
                merged_spans.append(self._create_span_with_comments(chunk))

            # 可以合并的块
            elif current_size + chunk_size <= self.chunk_size:
                current_group.append(chunk)
                current_size += chunk_size

            # 当前组已满，开始新组
            else:
                if current_group:
                    merged_spans.extend(self._merge_group(current_group))

                current_group = [chunk]
                current_size = chunk_size

        # 处理最后一组
        if current_group:
            merged_spans.extend(self._merge_group(current_group))

        return merged_spans

    def _create_documents(self, spans: List[Span], path: str, text: str) -> List[Document]:
        """生成最终的文档列表"""
        documents = []
        for span in spans:
            documents.append(Document(
                chunk_id=str(uuid.uuid4()),
                path=path,
                content=text[span.start:span.end],
                start_line=self._get_line_number(text, span.start),
                end_line=self._get_line_number(text, span.end)
            ))
        return documents

    def split(self, path: str, text: str) -> list[Document]:
        try:
            parser = get_parser_for_language(self.lang)
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

            # 阶段1: 解析所有语义块
            semantic_chunks = self._parse_semantic_chunks(root_node, text)

            # 如果没有找到语义块，降级到默认分割器
            if not semantic_chunks:
                logging.info(f"No semantic chunks found for {path}, falling back to default splitter")
                fallback_splitter = DefaultSplitter(self.chunk_size, self.chunk_overlap)
                return fallback_splitter.split(path, text)

            # 阶段2: 按优先级和大小合并
            merged_spans = self._merge_by_priority(semantic_chunks, text)

            # 生成最终文档
            return self._create_documents(merged_spans, path, text)

        except Exception as e:
            logging.error(f"Unexpected error parsing {path}: {e}, falling back to default splitter")
            fallback_splitter = DefaultSplitter(self.chunk_size, self.chunk_overlap)
            return fallback_splitter.split(path, text)


def get_splitter_parser(file_path: str) -> BaseSplitter:
    try:
        suffix = Path(file_path).suffix
        lang = settings.ext_to_lang.get(suffix)
        if lang:
            return CodeSplitter(chunk_size=2000, chunk_overlap=0, lang=lang)
    except Exception as e:
        logging.warning(f"Failed to get language for file: {file_path}: {e}")
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

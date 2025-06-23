"""
Semantic code splitter implementation using tree-sitter.
"""
import logging
import uuid
from typing import List, Optional

try:
    import tree_sitter
    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False

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

from .models import (
    Document, 
    SemanticChunk, 
    SemanticType, 
    Span, 
    SplitterConfig,
    SEMANTIC_TYPE_MAPPING
)
from .default_splitter import DefaultSplitter


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


class SemanticSplitter:
    """基于语义的代码分割器，使用tree-sitter进行AST解析"""
    
    def __init__(self, language: str, config: SplitterConfig = None):
        self.language = language
        self.config = config or SplitterConfig.for_code()
        self._fallback_splitter = DefaultSplitter(self.config)
    
    def split(self, path: str, text: str) -> List[Document]:
        """分割代码文本为语义块"""
        try:
            parser = get_parser_for_language(self.language)
            if parser is None:
                logging.warning(f"No parser available for language: {self.language}, falling back to default splitter")
                return self._fallback_splitter.split(path, text)

            tree = parser.parse(bytes(text, 'utf-8'))
            root_node = tree.root_node

            if not root_node or root_node.type == "ERROR":
                logging.warning(f"Parse error for {path}, falling back to default splitter")
                return self._fallback_splitter.split(path, text)

            # 阶段1: 解析所有语义块
            semantic_chunks = self._parse_semantic_chunks(root_node, text)

            # 如果没有找到语义块，降级到默认分割器
            if not semantic_chunks:
                logging.info(f"No semantic chunks found for {path}, falling back to default splitter")
                return self._fallback_splitter.split(path, text)

            # 阶段2: 按优先级和大小合并
            merged_spans = self._merge_by_priority(semantic_chunks, text)

            # 生成最终文档
            return self._create_documents(merged_spans, path, text)

        except Exception as e:
            logging.error(f"Unexpected error parsing {path}: {e}, falling back to default splitter")
            return self._fallback_splitter.split(path, text)
    
    def _get_semantic_type(self, node_type: str) -> SemanticType:
        """根据AST节点类型确定语义类型"""
        return SEMANTIC_TYPE_MAPPING.get(node_type, SemanticType.OTHER)
    
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
            if is_comment_node(node.type, self.language):
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

            if not between_text and is_documentation_comment(comment.type, self.language):
                associated_comments.append(comment)
            elif semantic_start - comment.end_byte < self.config.comment_distance_threshold:
                lines_between = text[comment.end_byte:semantic_start].count('\n')
                if lines_between <= self.config.max_lines_between_comment:
                    associated_comments.append(comment)

        return associated_comments
    
    def _is_semantic_boundary(self, node) -> bool:
        """Check if a node represents a semantic boundary (class, function, etc.)."""
        return is_semantic_boundary(node.type, self.language)

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
            if (chunk.semantic_type.priority >= self.config.high_priority_threshold and
                chunk_size > self.config.chunk_size * self.config.large_chunk_ratio):

                # 先处理当前组
                if current_group:
                    merged_spans.extend(self._merge_group(current_group))
                    current_group = []
                    current_size = 0

                # 高优先级块独立
                merged_spans.append(self._create_span_with_comments(chunk))

            # 可以合并的块
            elif current_size + chunk_size <= self.config.chunk_size:
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

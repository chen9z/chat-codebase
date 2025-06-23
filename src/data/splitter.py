import logging
import os
import sys
import uuid
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Any
from enum import Enum
from functools import lru_cache
import hashlib

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
    is_documentation_comment,
    DEFAULT_PARSER_CONFIG
)

if TREE_SITTER_AVAILABLE:
    from src.config.tree_sitter_config import get_parser, get_language
else:
    def get_parser(language: str):
        return None
    def get_language(language: str):
        return None

warnings.simplefilter(action='ignore', category=FutureWarning)

# 全局解析器缓存
_parser_cache: Dict[str, Any] = {}

@lru_cache(maxsize=128)
def get_parser_for_language(language: str):
    """Get a parser for the given language with caching."""
    cache_key = f"parser_{language}"
    
    if cache_key in _parser_cache:
        return _parser_cache[cache_key]
    
    try:
        # Try the new API first
        lang = get_language(language)
        if lang is not None:
            parser = lang.parser()
            _parser_cache[cache_key] = parser
            return parser
        else:
            # Fallback to old API
            parser = get_parser(language)
            if parser is not None:
                _parser_cache[cache_key] = parser
                return parser
            else:
                logging.warning(f"No parser available for language {language}")
                _parser_cache[cache_key] = None
                return None
    except Exception as e:
        logging.warning(f"Failed to create parser for language {language}: {e}")
        _parser_cache[cache_key] = None
        return None


@dataclass
class SplitterConfig:
    """分片器配置类，支持更细粒度的控制"""
    chunk_size: int = 2000
    chunk_overlap: int = 100
    
    # 语义分块配置
    min_semantic_chunk_size: int = 200  # 最小语义块大小
    max_semantic_chunk_size: int = 5000  # 最大语义块大小
    semantic_merge_threshold: float = 0.7  # 语义合并阈值
    
    # 注释关联配置
    max_comment_distance: int = 200  # 注释与代码的最大距离
    max_comment_lines_gap: int = 3  # 注释与代码间最大行数
    
    # 高优先级独立阈值
    high_priority_independence_ratio: float = 0.3  # 高优先级块独立的大小比例
    
    # 缓存配置
    enable_parsing_cache: bool = True
    cache_ttl_seconds: int = 300  # 缓存生存时间


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
    
    def size(self) -> int:
        return self.end - self.start
    
    def overlaps(self, other: 'Span') -> bool:
        return not (self.end <= other.start or other.end <= self.start)
    
    def merge(self, other: 'Span') -> 'Span':
        return Span(min(self.start, other.start), max(self.end, other.end))


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
    complexity_score: float = 0.0  # 复杂度评分，用于更智能的合并
    
    def __post_init__(self):
        if self.size == 0:
            self.size = self.span.size()


@dataclass
class Document:
    chunk_id: str = ""
    path: str = ""
    content: str = ""
    score: float = 0.0
    start_line: int = 0
    end_line: int = 0
    semantic_info: Dict = field(default_factory=dict)  # 新增语义信息字段


def get_content(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as file:
        return file.read()


# 文件内容缓存
_content_hash_cache: Dict[str, Tuple[str, List[Document]]] = {}

def get_file_hash(content: str) -> str:
    """计算文件内容的哈希值，用于缓存"""
    return hashlib.md5(content.encode()).hexdigest()


class BaseSplitter:
    def __init__(self, config: Optional[SplitterConfig] = None):
        self.config = config or SplitterConfig()

    def split(self, path: str, text: str) -> list[Document]:
        pass


class DefaultSplitter(BaseSplitter):
    def __init__(self, chunk_size: int = 2000, chunk_overlap: int = 100, config: Optional[SplitterConfig] = None):
        super().__init__(config)
        # 为了向后兼容，保留原有的参数
        if config is None:
            self.config.chunk_size = chunk_size
            self.config.chunk_overlap = chunk_overlap

    def split(self, path: str, text: str) -> list[Document]:
        # 使用缓存
        if self.config.enable_parsing_cache:
            file_hash = get_file_hash(text)
            cache_key = f"default_{file_hash}_{self.config.chunk_size}_{self.config.chunk_overlap}"
            
            if cache_key in _content_hash_cache:
                cached_hash, cached_result = _content_hash_cache[cache_key]
                if cached_hash == file_hash:
                    return cached_result

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
                if current_size + line_length > self.config.chunk_size and current_lines:
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
                        semantic_info={"splitter": "default", "chunk_type": "line_based"}
                    )
                )

                # Calculate overlap for next chunk - simplified logic
                if self.config.chunk_overlap > 0 and i < len(lines):
                    # Calculate how many lines to overlap based on character count
                    overlap_size = 0
                    overlap_lines = 0

                    # Start from the end and work backwards
                    for j in range(len(current_lines) - 1, -1, -1):
                        line = current_lines[j]
                        line_size = len(line) + 1  # +1 for newline

                        if overlap_size + line_size <= self.config.chunk_overlap:
                            overlap_size += line_size
                            overlap_lines += 1
                        else:
                            break

                    # Move back to create overlap, but ensure we don't go backwards
                    if overlap_lines > 0 and overlap_lines < len(current_lines):
                        i -= overlap_lines

        # 缓存结果
        if self.config.enable_parsing_cache:
            _content_hash_cache[cache_key] = (file_hash, chunks)

        return chunks


class CodeSplitter(BaseSplitter):
    def __init__(self, chunk_size: int, chunk_overlap: int, lang: str, config: Optional[SplitterConfig] = None):
        super().__init__(config)
        # 为了向后兼容，保留原有的参数
        if config is None:
            self.config.chunk_size = chunk_size
            self.config.chunk_overlap = chunk_overlap
        self.lang = lang

    def _calculate_complexity_score(self, node, text: str) -> float:
        """计算代码块的复杂度评分"""
        content = text[node.start_byte:node.end_byte]
        score = 0.0
        
        # 基于长度的基础分数
        score += len(content) / 1000.0
        
        # 基于嵌套级别
        nesting_level = content.count('{') + content.count('(') + content.count('[')
        score += nesting_level * 0.1
        
        # 基于关键字数量
        keywords = ['if', 'for', 'while', 'try', 'catch', 'switch', 'case']
        for keyword in keywords:
            score += content.count(keyword) * 0.2
        
        return score

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
        """为特定语义节点查找关联的注释，使用配置化的距离参数"""
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
            elif semantic_start - comment.end_byte < self.config.max_comment_distance:
                lines_between = text[comment.end_byte:semantic_start].count('\n')
                if lines_between <= self.config.max_comment_lines_gap:
                    associated_comments.append(comment)

        return associated_comments

    def _is_semantic_boundary(self, node) -> bool:
        """Check if a node represents a semantic boundary (class, function, etc.)."""
        return is_semantic_boundary(node.type, self.lang)

    def _parse_semantic_chunks(self, root_node, text: str) -> List[SemanticChunk]:
        """第一阶段：解析所有语义块，添加复杂度评分"""
        chunks = []
        comments = self._collect_comments_once(root_node)  # 一次性收集注释

        def traverse(node):
            if self._is_semantic_boundary(node):
                semantic_type = self._get_semantic_type(node.type)
                chunk_size = node.end_byte - node.start_byte
                complexity_score = self._calculate_complexity_score(node, text)

                chunk = SemanticChunk(
                    span=Span(node.start_byte, node.end_byte),
                    semantic_type=semantic_type,
                    node_type=node.type,
                    name=self._extract_name(node, text),
                    size=chunk_size,
                    complexity_score=complexity_score
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

    def _should_merge_chunks(self, chunk1: SemanticChunk, chunk2: SemanticChunk) -> bool:
        """智能判断两个语义块是否应该合并"""
        # 基于优先级差异
        priority_diff = abs(chunk1.semantic_type.priority - chunk2.semantic_type.priority)
        if priority_diff > 3:  # 优先级差异太大
            return False
        
        # 基于复杂度
        total_complexity = chunk1.complexity_score + chunk2.complexity_score
        if total_complexity > 5.0:  # 复杂度过高
            return False
        
        # 基于大小
        total_size = chunk1.size + chunk2.size
        if total_size > self.config.max_semantic_chunk_size:
            return False
        
        # 基于语义关系（简化版本）
        if (chunk1.semantic_type == SemanticType.IMPORT and 
            chunk2.semantic_type == SemanticType.IMPORT):
            return True  # import语句可以合并
        
        return True

    def _merge_group(self, group: List[SemanticChunk]) -> List[Span]:
        """合并一组语义块，支持更智能的合并策略"""
        if not group:
            return []

        # 按优先级排序，高优先级的放在前面
        group.sort(key=lambda c: (c.semantic_type.priority, c.complexity_score), reverse=True)

        # 检查是否可以智能合并
        if len(group) > 1:
            merged_groups = []
            current_subgroup = [group[0]]
            
            for i in range(1, len(group)):
                if self._should_merge_chunks(current_subgroup[-1], group[i]):
                    current_subgroup.append(group[i])
                else:
                    merged_groups.append(current_subgroup)
                    current_subgroup = [group[i]]
            
            merged_groups.append(current_subgroup)
            
            # 为每个子组创建span
            spans = []
            for subgroup in merged_groups:
                start = min(c.span.start for c in subgroup)
                end = max(c.span.end for c in subgroup)
                
                # 包含所有相关注释
                all_comments = []
                for chunk in subgroup:
                    if chunk.associated_comments:
                        all_comments.extend(chunk.associated_comments)

                if all_comments:
                    comment_start = min(c.start_byte for c in all_comments)
                    start = min(start, comment_start)
                
                spans.append(Span(start, end))
            
            return spans
        
        # 单个块的情况
        return [self._create_span_with_comments(group[0])]

    def _merge_by_priority(self, semantic_chunks: List[SemanticChunk], text: str) -> List[Span]:
        """第二阶段：按优先级和大小智能合并，支持配置化阈值"""
        if not semantic_chunks:
            return []

        merged_spans = []
        current_group = []
        current_size = 0

        for chunk in semantic_chunks:
            chunk_size = chunk.size

            # 检查大小限制
            if chunk_size < self.config.min_semantic_chunk_size:
                # 小块强制合并
                current_group.append(chunk)
                current_size += chunk_size
                continue

            # 高优先级的大块独立成chunk，使用配置化阈值
            if (chunk.semantic_type.priority >= 8 and
                chunk_size > self.config.chunk_size * self.config.high_priority_independence_ratio):

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

    def _create_documents(self, spans: List[Span], path: str, text: str, semantic_chunks: List[SemanticChunk]) -> List[Document]:
        """生成最终的文档列表，添加语义信息"""
        documents = []
        
        for span in spans:
            # 收集这个span包含的语义信息
            contained_chunks = [chunk for chunk in semantic_chunks 
                              if chunk.span.start >= span.start and chunk.span.end <= span.end]
            
            semantic_info = {
                "splitter": "semantic",
                "chunk_types": [chunk.semantic_type.type_name for chunk in contained_chunks],
                "chunk_names": [chunk.name for chunk in contained_chunks if chunk.name],
                "complexity_scores": [chunk.complexity_score for chunk in contained_chunks],
                "has_comments": any(chunk.associated_comments for chunk in contained_chunks)
            }
            
            documents.append(Document(
                chunk_id=str(uuid.uuid4()),
                path=path,
                content=text[span.start:span.end],
                start_line=self._get_line_number(text, span.start),
                end_line=self._get_line_number(text, span.end),
                semantic_info=semantic_info
            ))
        return documents

    def split(self, path: str, text: str) -> list[Document]:
        # 使用缓存
        if self.config.enable_parsing_cache:
            file_hash = get_file_hash(text)
            cache_key = f"semantic_{self.lang}_{file_hash}_{self.config.chunk_size}"
            
            if cache_key in _content_hash_cache:
                cached_hash, cached_result = _content_hash_cache[cache_key]
                if cached_hash == file_hash:
                    return cached_result

        try:
            parser = get_parser_for_language(self.lang)
            if parser is None:
                logging.warning(f"No parser available for language: {self.lang}, falling back to default splitter")
                fallback_splitter = DefaultSplitter(config=self.config)
                return fallback_splitter.split(path, text)

            tree = parser.parse(bytes(text, 'utf-8'))
            root_node = tree.root_node

            if not root_node or root_node.type == "ERROR":
                logging.warning(f"Parse error for {path}, falling back to default splitter")
                fallback_splitter = DefaultSplitter(config=self.config)
                return fallback_splitter.split(path, text)

            # 阶段1: 解析所有语义块
            semantic_chunks = self._parse_semantic_chunks(root_node, text)

            # 如果没有找到语义块，降级到默认分割器
            if not semantic_chunks:
                logging.info(f"No semantic chunks found for {path}, falling back to default splitter")
                fallback_splitter = DefaultSplitter(config=self.config)
                return fallback_splitter.split(path, text)

            # 阶段2: 按优先级和大小合并
            merged_spans = self._merge_by_priority(semantic_chunks, text)

            # 生成最终文档
            result = self._create_documents(merged_spans, path, text, semantic_chunks)

            # 缓存结果
            if self.config.enable_parsing_cache:
                _content_hash_cache[cache_key] = (file_hash, result)

            return result

        except Exception as e:
            logging.error(f"Unexpected error parsing {path}: {e}, falling back to default splitter")
            fallback_splitter = DefaultSplitter(config=self.config)
            return fallback_splitter.split(path, text)


def get_splitter_parser(file_path: str, config: Optional[SplitterConfig] = None, use_language_specific: bool = True) -> BaseSplitter:
    """获取分片解析器，支持配置化和语言特定处理"""
    try:
        suffix = Path(file_path).suffix
        lang = settings.ext_to_lang.get(suffix)
        
        if lang and use_language_specific:
            # 尝试使用语言特定的分片器
            try:
                from .language_specific_splitters import get_language_specific_splitter
                return get_language_specific_splitter(lang, config)
            except ImportError:
                logging.warning("Language specific splitters not available, falling back to generic CodeSplitter")
                return CodeSplitter(chunk_size=2000, chunk_overlap=0, lang=lang, config=config)
        elif lang:
            # 使用通用的代码分片器
            return CodeSplitter(chunk_size=2000, chunk_overlap=0, lang=lang, config=config)
    except Exception as e:
        logging.warning(f"Failed to get language for file: {file_path}: {e}")
    
    return DefaultSplitter(chunk_size=2000, chunk_overlap=100, config=config)


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

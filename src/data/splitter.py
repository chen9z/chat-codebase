import logging
import os
import sys
import uuid
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Tuple
from enum import Enum
from bisect import bisect_left

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
    """语义类型枚举，包含类型名称、优先级和理想大小"""
    # 高优先级 - 独立性强的语义单元
    CLASS = ("class", 10, 1000)  # (type_name, priority, ideal_size)
    INTERFACE = ("interface", 10, 800)
    ENUM = ("enum", 9, 600)
    FUNCTION = ("function", 8, 800)
    METHOD = ("method", 8, 600)

    # 中优先级 - 相关的辅助内容
    CONSTRUCTOR = ("constructor", 6, 400)
    PROPERTY = ("property", 5, 200)
    COMMENT_BLOCK = ("comment_block", 4, 300)

    # 低优先级 - 可合并的内容
    IMPORT = ("import", 2, 100)
    VARIABLE = ("variable", 1, 100)
    OTHER = ("other", 0, 200)

    def __init__(self, type_name, priority, ideal_size):
        self.type_name = type_name
        self.priority = priority
        self.ideal_size = ideal_size


@dataclass
class Span:
    start: int
    end: int
    
    @property
    def size(self) -> int:
        return self.end - self.start


@dataclass
class SemanticChunk:
    """语义块，包含语义信息的代码片段"""
    span: Span
    semantic_type: SemanticType
    node_type: str  # 原始 AST 节点类型
    name: str = ""  # 提取的名称（类名、函数名等）
    associated_comments: List = field(default_factory=list)
    can_merge: bool = True
    complexity_score: float = 1.0  # 复杂度评分，影响合并决策

    @property
    def size(self) -> int:
        return self.span.size

    @property
    def effective_priority(self) -> float:
        """计算有效优先级，结合语义优先级和复杂度"""
        return self.semantic_type.priority + (self.complexity_score * 0.5)


@dataclass
class Document:
    chunk_id: str = ""
    path: str = ""
    content: str = ""
    score: float = 0.0
    start_line: int = 0
    end_line: int = 0
    semantic_info: Dict = field(default_factory=dict)  # 添加语义信息


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
                        semantic_info={"type": "text_chunk", "method": "default"}
                    )
                )

                # Calculate overlap for next chunk - optimized logic
                if self.chunk_overlap > 0 and i < len(lines):
                    overlap_lines = self._calculate_overlap_lines(current_lines)
                    if overlap_lines > 0:
                        i -= overlap_lines

        return chunks

    def _calculate_overlap_lines(self, current_lines: List[str]) -> int:
        """计算重叠行数的优化方法"""
        overlap_size = 0
        overlap_lines = 0
        
        # 从末尾向前计算重叠
        for j in range(len(current_lines) - 1, -1, -1):
            line = current_lines[j]
            line_size = len(line) + 1  # +1 for newline

            if overlap_size + line_size <= self.chunk_overlap:
                overlap_size += line_size
                overlap_lines += 1
            else:
                break

        # 确保不会超过当前块的行数
        return min(overlap_lines, len(current_lines) - 1)


class CodeSplitter(BaseSplitter):
    def __init__(self, chunk_size: int, chunk_overlap: int, lang: str):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.lang = lang
        self._language_config = self._get_language_config(lang)

    def _get_language_config(self, lang: str) -> Dict:
        """获取语言特定的配置"""
        configs = {
            'python': {
                'class_keywords': ['class'],
                'function_keywords': ['def', 'async def'],
                'complexity_indicators': ['if', 'for', 'while', 'try', 'with'],
                'max_function_size': 1000,
            },
            'javascript': {
                'class_keywords': ['class'],
                'function_keywords': ['function', 'const', 'let', 'var'],
                'complexity_indicators': ['if', 'for', 'while', 'try', 'switch'],
                'max_function_size': 800,
            },
            'java': {
                'class_keywords': ['class', 'interface', 'enum'],
                'function_keywords': ['public', 'private', 'protected'],
                'complexity_indicators': ['if', 'for', 'while', 'try', 'switch'],
                'max_function_size': 1200,
            },
        }
        return configs.get(lang, configs['python'])

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

    def _calculate_complexity_score(self, node, text: str) -> float:
        """计算代码复杂度评分"""
        try:
            node_text = text[node.start_byte:node.end_byte]
            complexity_indicators = self._language_config.get('complexity_indicators', [])
            
            score = 1.0
            for indicator in complexity_indicators:
                score += node_text.count(indicator) * 0.1
            
            # 根据代码长度调整
            if len(node_text) > 500:
                score += 0.5
            if len(node_text) > 1000:
                score += 0.5
                
            return min(score, 3.0)  # 限制最大复杂度
        except:
            return 1.0

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
        """为特定语义节点查找关联的注释 - 使用二分查找优化"""
        associated_comments = []
        semantic_start = semantic_node.start_byte
        
        # 使用二分查找找到第一个可能相关的注释
        comment_starts = [c.start_byte for c in all_comments]
        insert_pos = bisect_left(comment_starts, semantic_start)
        
        # 只检查语义节点之前的注释
        for i in range(min(insert_pos, len(all_comments))):
            comment = all_comments[i]
            
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
        comments = self._collect_comments_once(root_node)

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
                    complexity_score=complexity_score
                )

                # 关联注释
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

    def _should_merge_chunks(self, chunk1: SemanticChunk, chunk2: SemanticChunk, 
                           current_size: int) -> bool:
        """判断两个语义块是否应该合并"""
        # 检查大小限制
        if current_size + chunk2.size > self.chunk_size:
            return False
        
        # 高优先级的大块通常不合并
        if (chunk2.effective_priority >= 8 and 
            chunk2.size > chunk2.semantic_type.ideal_size):
            return False
            
        # 类型相似的更容易合并
        type_compatibility = {
            (SemanticType.FUNCTION, SemanticType.METHOD): True,
            (SemanticType.IMPORT, SemanticType.VARIABLE): True,
            (SemanticType.PROPERTY, SemanticType.VARIABLE): True,
        }
        
        chunk_types = (chunk1.semantic_type, chunk2.semantic_type)
        if chunk_types in type_compatibility or chunk_types[::-1] in type_compatibility:
            return True
            
        # 同类型的小块容易合并
        if (chunk1.semantic_type == chunk2.semantic_type and 
            chunk2.size < chunk2.semantic_type.ideal_size * 0.5):
            return True
            
        return chunk2.effective_priority < 5  # 低优先级的可以合并

    def _merge_group(self, group: List[SemanticChunk]) -> List[Span]:
        """合并一组语义块"""
        if not group:
            return []

        # 按优先级排序，高优先级的放在前面
        group.sort(key=lambda c: c.effective_priority, reverse=True)

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
        """第二阶段：按优先级和大小智能合并 - 优化版本"""
        if not semantic_chunks:
            return []

        merged_spans = []
        current_group = []
        current_size = 0

        for i, chunk in enumerate(semantic_chunks):
            chunk_size = chunk.size

            # 检查是否应该独立成块
            should_be_independent = (
                chunk.effective_priority >= 8 and 
                chunk_size > self.chunk_size * 0.3
            )

            # 检查是否可以与当前组合并
            can_merge_with_current = (
                current_group and 
                self._should_merge_chunks(current_group[-1], chunk, current_size)
            )

            if should_be_independent:
                # 先处理当前组
                if current_group:
                    merged_spans.extend(self._merge_group(current_group))
                    current_group = []
                    current_size = 0

                # 高优先级块独立
                merged_spans.append(self._create_span_with_comments(chunk))

            elif can_merge_with_current:
                # 与当前组合并
                current_group.append(chunk)
                current_size += chunk_size

            else:
                # 开始新组
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
        for i, span in enumerate(spans):
            content = text[span.start:span.end]
            
            # 分析内容的语义信息
            semantic_info = self._analyze_chunk_semantics(content)
            semantic_info.update({
                "method": "semantic",
                "chunk_index": i,
                "total_chunks": len(spans)
            })
            
            documents.append(Document(
                chunk_id=str(uuid.uuid4()),
                path=path,
                content=content,
                start_line=self._get_line_number(text, span.start),
                end_line=self._get_line_number(text, span.end),
                semantic_info=semantic_info
            ))
        return documents

    def _analyze_chunk_semantics(self, content: str) -> Dict:
        """分析代码块的语义信息"""
        info = {
            "has_classes": False,
            "has_functions": False,
            "has_imports": False,
            "has_comments": False,
            "estimated_complexity": "low"
        }
        
        # 简单的关键词检测
        class_keywords = self._language_config.get('class_keywords', [])
        function_keywords = self._language_config.get('function_keywords', [])
        complexity_indicators = self._language_config.get('complexity_indicators', [])
        
        for keyword in class_keywords:
            if keyword in content:
                info["has_classes"] = True
                break
                
        for keyword in function_keywords:
            if keyword in content:
                info["has_functions"] = True
                break
                
        if any(keyword in content for keyword in ["import", "from", "include", "require"]):
            info["has_imports"] = True
            
        if any(indicator in content for indicator in ["#", "//", "/*", '"""', "'''"]):
            info["has_comments"] = True
            
        # 估算复杂度
        complexity_count = sum(content.count(indicator) for indicator in complexity_indicators)
        if complexity_count > 10:
            info["estimated_complexity"] = "high"
        elif complexity_count > 5:
            info["estimated_complexity"] = "medium"
            
        return info

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
            documents = self._create_documents(merged_spans, path, text)
            
            logging.info(f"Split {path} into {len(documents)} semantic chunks")
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
            return CodeSplitter(chunk_size=2000, chunk_overlap=0, lang=lang)
    except Exception as e:
        logging.warning(f"Failed to get language for file: {file_path}: {e}")
    return DefaultSplitter(chunk_size=2000, chunk_overlap=100)


def parse(file_path: str) -> list[Document]:
    """Parse a file and return a list of documents"""
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
            content = file.read()
        
        splitter = get_splitter_parser(file_path)
        return splitter.split(file_path, content)
    except Exception as e:
        logging.error(f"Failed to parse file {file_path}: {e}")
        return []


if __name__ == "__main__":
    java_path = os.path.expanduser(
        "~/workspace/spring-ai/models/spring-ai-qianfan/src/main/java/org/springframework/ai/qianfan/api/QianFanApi.java")
    documents = parse(java_path)
    for doc in documents:
        print(f"===============content length: {len(doc.content)}")
        print(doc.content)

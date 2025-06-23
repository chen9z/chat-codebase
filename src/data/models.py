"""
Data models and enums for the splitter module.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import List


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
    """表示文本中的一个范围"""
    start: int
    end: int

    @property
    def size(self) -> int:
        """返回span的大小"""
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

    @property
    def size(self) -> int:
        """返回语义块的大小"""
        return self.span.size


@dataclass
class Document:
    """文档块，表示最终的分割结果"""
    chunk_id: str = ""
    path: str = ""
    content: str = ""
    score: float = 0.0
    start_line: int = 0
    end_line: int = 0


# 语义类型映射配置
SEMANTIC_TYPE_MAPPING = {
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
    "function_declaration": SemanticType.FUNCTION,
    "method_definition": SemanticType.METHOD,
    "arrow_function": SemanticType.FUNCTION,
    "function_expression": SemanticType.FUNCTION,
    "generator_function_declaration": SemanticType.FUNCTION,
    "type_alias_declaration": SemanticType.INTERFACE,

    # Go
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


class SplitterConfig:
    """分割器配置类"""
    
    def __init__(
        self,
        chunk_size: int = 2000,
        chunk_overlap: int = 100,
        high_priority_threshold: int = 8,
        large_chunk_ratio: float = 0.3,
        comment_distance_threshold: int = 200,
        max_lines_between_comment: int = 2
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.high_priority_threshold = high_priority_threshold
        self.large_chunk_ratio = large_chunk_ratio
        self.comment_distance_threshold = comment_distance_threshold
        self.max_lines_between_comment = max_lines_between_comment
    
    @classmethod
    def default(cls) -> 'SplitterConfig':
        """返回默认配置"""
        return cls()
    
    @classmethod
    def for_code(cls) -> 'SplitterConfig':
        """返回代码分割的配置"""
        return cls(chunk_size=2000, chunk_overlap=0)
    
    @classmethod
    def for_text(cls) -> 'SplitterConfig':
        """返回文本分割的配置"""
        return cls(chunk_size=2000, chunk_overlap=100)

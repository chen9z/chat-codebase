"""
Configuration for code parsing and syntax analysis.
"""

from dataclasses import dataclass
from typing import Dict, Set, Optional


@dataclass
class ParserConfig:
    """Configuration for code parsing."""
    
    # Default chunk sizes
    default_chunk_size: int = 2000
    default_chunk_overlap: int = 100
    
    # Maximum chunk size multiplier for semantic boundaries
    semantic_boundary_multiplier: float = 1.5
    
    # Cache settings
    enable_parser_cache: bool = True
    max_cache_size: int = 100
    
    # Error handling
    fallback_to_default_splitter: bool = True
    log_parse_errors: bool = True


# Language-specific comment node types
COMMENT_NODE_TYPES: Dict[str, Set[str]] = {
    "java": {
        "line_comment",
        "block_comment",
        "javadoc_comment",
    },
    "python": {
        "comment",
    },
    "javascript": {
        "comment",
        "line_comment",
        "block_comment",
    },
    "typescript": {
        "comment",
        "line_comment",
        "block_comment",
    },
    "go": {
        "comment",
        "line_comment",
        "block_comment",
    },
    "rust": {
        "line_comment",
        "block_comment",
        "doc_comment",
    },
    "cpp": {
        "comment",
        "line_comment",
        "block_comment",
    },
    "c": {
        "comment",
        "line_comment",
        "block_comment",
    },
    "csharp": {
        "comment",
        "line_comment",
        "block_comment",
        "documentation_comment",
    },
    "php": {
        "comment",
        "line_comment",
        "block_comment",
    },
    "ruby": {
        "comment",
        "line_comment",
        "block_comment",
    },
    "swift": {
        "comment",
        "line_comment",
        "block_comment",
    },
    "kotlin": {
        "comment",
        "line_comment",
        "block_comment",
    },
    "scala": {
        "comment",
        "line_comment",
        "block_comment",
    },
}

# Language-specific semantic node types
SEMANTIC_NODE_TYPES: Dict[str, Set[str]] = {
    "python": {
        "class_definition",
        "function_definition",
        "async_function_definition",
        "decorated_definition",
    },
    "java": {
        "class_declaration",
        "interface_declaration",
        "method_declaration",
        "constructor_declaration",
        "enum_declaration",
        "annotation_type_declaration",
        "record_declaration",
    },
    "javascript": {
        "class_declaration",
        "function_declaration",
        "method_definition",
        "arrow_function",
        "function_expression",
        "generator_function_declaration",
    },
    "typescript": {
        "class_declaration",
        "interface_declaration",
        "function_declaration",
        "method_definition",
        "arrow_function",
        "function_expression",
        "type_alias_declaration",
        "enum_declaration",
    },
    "go": {
        "function_declaration",
        "method_declaration",
        "type_declaration",
        "interface_type",
        "struct_type",
    },
    "rust": {
        "function_item",
        "impl_item",
        "struct_item",
        "enum_item",
        "trait_item",
        "mod_item",
        "associated_type",
    },
    "cpp": {
        "function_definition",
        "class_specifier",
        "struct_specifier",
        "namespace_definition",
        "template_declaration",
        "union_specifier",
        "enum_specifier",
    },
    "c": {
        "function_definition",
        "struct_specifier",
        "union_specifier",
        "enum_specifier",
    },
    "csharp": {
        "class_declaration",
        "interface_declaration",
        "method_declaration",
        "constructor_declaration",
        "enum_declaration",
        "struct_declaration",
        "namespace_declaration",
    },
    "php": {
        "class_declaration",
        "interface_declaration",
        "function_definition",
        "method_declaration",
        "trait_declaration",
    },
    "ruby": {
        "class",
        "module",
        "method",
        "singleton_method",
    },
    "swift": {
        "class_declaration",
        "struct_declaration",
        "protocol_declaration",
        "function_declaration",
        "enum_declaration",
    },
    "kotlin": {
        "class_declaration",
        "interface_declaration",
        "function_declaration",
        "object_declaration",
    },
    "scala": {
        "class_definition",
        "object_definition",
        "trait_definition",
        "function_definition",
    },
}

# Default configuration instance
DEFAULT_PARSER_CONFIG = ParserConfig()


def get_semantic_types(language: str) -> Set[str]:
    """Get semantic node types for a specific language."""
    return SEMANTIC_NODE_TYPES.get(language, set())


def get_comment_types(language: str) -> Set[str]:
    """Get comment node types for a specific language."""
    return COMMENT_NODE_TYPES.get(language, set())


def is_comment_node(node_type: str, language: str) -> bool:
    """Check if a node type represents a comment for the given language."""
    comment_types = get_comment_types(language)
    return node_type in comment_types


def is_semantic_boundary(node_type: str, language: str) -> bool:
    """Check if a node type represents a semantic boundary for the given language."""
    semantic_types = get_semantic_types(language)
    return node_type in semantic_types


def is_documentation_comment(node_type: str, language: str) -> bool:
    """Check if a node type represents a documentation comment (like Javadoc)."""
    doc_comment_types = {
        "java": {"javadoc_comment", "block_comment"},
        "python": {"comment"},  # Python docstrings are actually string literals
        "javascript": {"block_comment"},
        "typescript": {"block_comment"},
        "rust": {"doc_comment"},
        "cpp": {"block_comment"},
        "c": {"block_comment"},
        "csharp": {"documentation_comment", "block_comment"},
        "php": {"block_comment"},
        "ruby": {"block_comment"},
        "swift": {"block_comment"},
        "kotlin": {"block_comment"},
        "scala": {"block_comment"},
    }
    return node_type in doc_comment_types.get(language, set())

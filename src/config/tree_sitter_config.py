"""
Tree-sitter language configuration and parser management.
"""
import logging
import os
from typing import Dict, Optional

try:
    import tree_sitter
    from tree_sitter import Language, Parser
    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False
    logging.warning("tree-sitter not available, falling back to default splitter")


# Language mapping for tree-sitter
LANGUAGE_MAPPING = {
    'python': 'python',
    'java': 'java',
    'javascript': 'javascript',
    'typescript': 'typescript',
    'go': 'go',
    'rust': 'rust',
    'cpp': 'cpp',
    'c': 'c',
    'csharp': 'c_sharp',
    'php': 'php',
    'ruby': 'ruby',
    'swift': 'swift',
    'kotlin': 'kotlin',
    'scala': 'scala',
}

# Cache for loaded languages and parsers
_language_cache: Dict[str, Language] = {}
_parser_cache: Dict[str, Parser] = {}


def _try_load_language(language_name: str) -> Optional[Language]:
    """尝试加载tree-sitter语言库"""
    if not TREE_SITTER_AVAILABLE:
        return None

    try:
        # 尝试从tree_sitter_languages包加载（如果可用）
        try:
            import tree_sitter_languages
            return tree_sitter_languages.get_language(language_name)
        except ImportError:
            pass

        # 尝试直接导入语言模块
        try:
            module_name = f"tree_sitter_{language_name}"
            language_module = __import__(module_name)
            return language_module.language()
        except ImportError:
            pass

        logging.debug(f"No tree-sitter grammar found for {language_name}")
        return None

    except Exception as e:
        logging.debug(f"Failed to load tree-sitter language {language_name}: {e}")
        return None


def get_language(language: str) -> Optional[Language]:
    """Get a tree-sitter language for the given language."""
    if not TREE_SITTER_AVAILABLE:
        return None

    # 标准化语言名称
    normalized_lang = LANGUAGE_MAPPING.get(language.lower(), language.lower())

    # 检查缓存
    if normalized_lang in _language_cache:
        return _language_cache[normalized_lang]

    # 尝试加载语言
    lang = _try_load_language(normalized_lang)
    if lang:
        _language_cache[normalized_lang] = lang
        logging.debug(f"Loaded tree-sitter language for {language}")
    else:
        logging.debug(f"No tree-sitter language available for {language}")

    return lang


def get_parser(language: str) -> Optional[Parser]:
    """Get a tree-sitter parser for the given language."""
    if not TREE_SITTER_AVAILABLE:
        return None

    # 标准化语言名称
    normalized_lang = LANGUAGE_MAPPING.get(language.lower(), language.lower())

    # 检查缓存
    if normalized_lang in _parser_cache:
        return _parser_cache[normalized_lang]

    # 获取语言
    lang = get_language(language)
    if not lang:
        return None

    # 创建解析器
    try:
        parser = Parser()
        parser.set_language(lang)
        _parser_cache[normalized_lang] = parser
        logging.debug(f"Created tree-sitter parser for {language}")
        return parser
    except Exception as e:
        logging.debug(f"Failed to create parser for {language}: {e}")
        return None

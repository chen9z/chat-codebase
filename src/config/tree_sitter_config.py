"""
Tree-sitter language configuration and parser management.
"""
import logging
from typing import Dict, Optional

try:
    import tree_sitter
    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False
    logging.warning("tree-sitter not available, falling back to default splitter")


def get_parser(language: str) -> Optional:
    """Get a tree-sitter parser for the given language."""
    if not TREE_SITTER_AVAILABLE:
        logging.warning(f"tree-sitter not available for language: {language}")
        return None

    # For now, return None to fall back to default splitter
    # This can be extended later when tree-sitter grammars are properly set up
    logging.info(f"tree-sitter parser not configured for language: {language}, falling back to default splitter")
    return None


def get_language(language: str) -> Optional:
    """Get a tree-sitter language for the given language."""
    if not TREE_SITTER_AVAILABLE:
        logging.warning(f"tree-sitter not available for language: {language}")
        return None

    # For now, return None to fall back to default splitter
    # This can be extended later when tree-sitter grammars are properly set up
    logging.info(f"tree-sitter language not configured for language: {language}, falling back to default splitter")
    return None

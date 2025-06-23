"""
Main splitter module providing unified interface for text and code splitting.

This module provides the main entry point for all text and code splitting functionality.
It automatically selects the appropriate splitter based on file type and provides
a clean, unified API for document chunking.

Recent Changes:
- Implemented two-phase semantic chunking architecture
- Optimized performance from O(n²) to O(n log n)
- Added modular design with separate splitter classes
- Improved comment association and semantic boundary detection
"""
import logging
from pathlib import Path
from typing import List, Protocol

from src.config import settings
from .models import Document, SplitterConfig
from .default_splitter import DefaultSplitter
from .semantic_splitter import SemanticSplitter


class BaseSplitter(Protocol):
    """Base protocol for all splitters"""
    
    def split(self, path: str, text: str) -> List[Document]:
        """Split text into documents"""
        ...


def get_content(path: str) -> str:
    """Read file content with error handling"""
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as file:
            return file.read()
    except Exception as e:
        logging.error(f"Failed to read file {path}: {e}")
        return ""


def get_splitter(file_path: str, config: SplitterConfig = None) -> BaseSplitter:
    """
    Get appropriate splitter for the given file.
    
    Args:
        file_path: Path to the file
        config: Optional configuration for the splitter
        
    Returns:
        Appropriate splitter instance
    """
    try:
        suffix = Path(file_path).suffix.lower()
        language = settings.ext_to_lang.get(suffix)
        
        if language:
            # Use semantic splitter for supported languages
            return SemanticSplitter(language, config or SplitterConfig.for_code())
        else:
            # Use default splitter for unsupported file types
            return DefaultSplitter(config or SplitterConfig.for_text())
            
    except Exception as e:
        logging.warning(f"Failed to determine splitter for file: {file_path}: {e}")
        return DefaultSplitter(config or SplitterConfig.default())


def parse(file_path: str, config: SplitterConfig = None) -> List[Document]:
    """
    Parse a file into document chunks.
    
    Args:
        file_path: Path to the file to parse
        config: Optional configuration for splitting
        
    Returns:
        List of document chunks
    """
    try:
        splitter = get_splitter(file_path, config)
        text = get_content(file_path)
        
        if not text:
            logging.warning(f"Empty or unreadable file: {file_path}")
            return []
            
        return splitter.split(file_path, text)
        
    except Exception as e:
        logging.error(f"Failed to parse file {file_path}: {e}")
        return []


# Backward compatibility aliases
def get_splitter_parser(file_path: str) -> BaseSplitter:
    """Backward compatibility alias for get_splitter"""
    return get_splitter(file_path)


class CodeSplitter(SemanticSplitter):
    """Backward compatibility alias for SemanticSplitter"""
    
    def __init__(self, chunk_size: int, chunk_overlap: int, lang: str):
        config = SplitterConfig(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        super().__init__(lang, config)


# For testing and debugging, use: python -c "from src.data.splitter import parse; ..."

"""
Default text splitter implementation.
"""
import uuid
from typing import List

from .models import Document, SplitterConfig


class DefaultSplitter:
    """默认的文本分割器，基于字符数和行数进行分割"""
    
    def __init__(self, config: SplitterConfig = None):
        self.config = config or SplitterConfig.for_text()
    
    def split(self, path: str, text: str) -> List[Document]:
        """将文本分割成多个文档块"""
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
                    )
                )

                # Calculate overlap for next chunk
                if self.config.chunk_overlap > 0 and i < len(lines):
                    overlap_lines = self._calculate_overlap(current_lines)
                    if overlap_lines > 0:
                        i -= overlap_lines

        return chunks
    
    def _calculate_overlap(self, current_lines: List[str]) -> int:
        """计算重叠的行数"""
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

        # Ensure we don't overlap the entire chunk
        return min(overlap_lines, len(current_lines) - 1) if overlap_lines > 0 else 0

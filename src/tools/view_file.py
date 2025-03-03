from typing import Any, Dict, List, Optional
from pathlib import Path

from .base import BaseTool

class ViewFileTool(BaseTool):
    """Tool for viewing file contents with line range support."""

    @property
    def name(self) -> str:
        return "view_file"

    @property
    def description(self) -> str:
        return """View the contents of a file. The lines of the file are 0-indexed,
        and the output will be the file contents from StartLine to EndLine,
        together with a summary of the lines outside of StartLine and EndLine."""

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to file to view"
                },
                "start_line": {
                    "type": "integer",
                    "description": "Start line to view (0-indexed)"
                },
                "end_line": {
                    "type": "integer",
                    "description": "End line to view (0-indexed)"
                }
            },
            "required": ["file_path", "start_line", "end_line"]
        }

    async def execute(
        self,
        file_path: str,
        start_line: int,
        end_line: int
    ) -> Dict[str, Any]:
        """View file contents within specified line range.
        
        Args:
            file_path: Path to the file to view
            start_line: Start line number (0-indexed)
            end_line: End line number (0-indexed)
            
        Returns:
            Dict containing file contents and metadata
        """
        try:
            path = Path(file_path)
            if not path.is_file():
                return {
                    "error": f"Path {file_path} is not a file",
                    "contents": None
                }

            # Ensure we don't try to read more than 200 lines at once
            if end_line - start_line > 200:
                end_line = start_line + 200

            with path.open('r', encoding='utf-8') as f:
                all_lines = f.readlines()
                total_lines = len(all_lines)

                # Validate line range
                start_line = max(0, min(start_line, total_lines - 1))
                end_line = max(0, min(end_line, total_lines - 1))
                
                if start_line > end_line:
                    start_line, end_line = end_line, start_line

                # Get the requested lines
                requested_lines = all_lines[start_line:end_line + 1]
                
                # Create summary of lines before and after
                before_summary = None
                after_summary = None
                
                if start_line > 0:
                    before_lines = min(5, start_line)  # Show up to 5 lines before
                    before_summary = {
                        "lines_not_shown": start_line - before_lines,
                        "preview": all_lines[start_line - before_lines:start_line]
                    }
                
                if end_line < total_lines - 1:
                    after_lines = min(5, total_lines - end_line - 1)  # Show up to 5 lines after
                    after_summary = {
                        "lines_not_shown": total_lines - end_line - 1 - after_lines,
                        "preview": all_lines[end_line + 1:end_line + 1 + after_lines]
                    }

                return {
                    "file": str(path),
                    "total_lines": total_lines,
                    "start_line": start_line,
                    "end_line": end_line,
                    "contents": requested_lines,
                    "before_summary": before_summary,
                    "after_summary": after_summary
                }

        except UnicodeDecodeError:
            return {
                "error": f"File {file_path} is not a text file or has unknown encoding",
                "contents": None
            }
        except Exception as e:
            return {
                "error": f"Error reading file {file_path}: {str(e)}",
                "contents": None
            } 
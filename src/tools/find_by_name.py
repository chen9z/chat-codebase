from typing import Any, Dict, List, Optional
import os
from pathlib import Path
from datetime import datetime
from fnmatch import fnmatch
import asyncio
import json

from src.tools.base import BaseTool  # Try relative import first


class FindByNameTool(BaseTool):
    """Tool for finding files and directories using glob patterns."""

    @property
    def name(self) -> str:
        return "find_by_name"

    @property
    def description(self) -> str:
        return """Search for files and directories within a specified directory,
        similar to the Linux find command. Supports glob patterns for searching
        and filtering."""

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "properties": {
                "search_directory": {
                    "type": "string",
                    "description": "The directory to search within"
                },
                "pattern": {
                    "type": "string",
                    "description": "Pattern to search for"
                },
                "includes": {
                    "items": {
                        "type": "string"
                    },
                    "type": "array",
                    "description": "Optional patterns to include"
                },
                "excludes": {
                    "items": {
                        "type": "string"
                    },
                    "type": "array",
                    "description": "Optional patterns to exclude"
                },
                "type": {
                    "type": "string",
                    "enum": ["file"],
                    "description": "Type filter (file)"
                },
                "max_depth": {
                    "type": "integer",
                    "description": "Maximum depth to search"
                }
            },
            "required": ["search_directory", "pattern"]
        }

    async def execute(
            self,
            search_directory: str,
            pattern: str,
            includes: Optional[List[str]] = None,
            excludes: Optional[List[str]] = None,
            type: Optional[str] = None,
            max_depth: Optional[int] = None
    ) -> Dict[str, Any]:
        """Execute file search with pattern matching.
        
        Args:
            search_directory: Base directory to search in
            pattern: Main search pattern
            includes: Additional patterns to include
            excludes: Patterns to exclude
            type: Type filter ('file' only)
            max_depth: Maximum directory depth to search
            
        Returns:
            Dict containing search results with file information
        """
        try:
            base_path = Path(search_directory).resolve()  # Get absolute path
            print(f"Searching in directory: {base_path}")
            if not base_path.is_dir():
                return {
                    "error": f"Search directory {search_directory} is not a directory",
                    "matches": []
                }

            matches = []
            for root, dirs, files in os.walk(base_path):
                # Print debug info
                print(f"Scanning directory: {root}")
                print(f"Found files: {files}")

                # Check depth limit
                if max_depth is not None:
                    current_depth = len(Path(root).relative_to(base_path).parts)
                    if current_depth > max_depth:
                        dirs.clear()  # Stop descending
                        continue

                # Process files
                items = files if type == "file" else files + dirs
                for item in items:
                    item_path = Path(root) / item
                    rel_path = item_path.relative_to(base_path)
                    str_path = str(rel_path)

                    # Print debug info
                    print(f"Checking file: {str_path}")
                    print(f"Against pattern: {pattern}")

                    # Apply pattern matching
                    if not fnmatch(str_path, pattern):
                        print(f"Pattern did not match")
                        continue

                    # Check includes
                    if includes and not any(fnmatch(str_path, inc) for inc in includes):
                        print(f"Not in includes list")
                        continue

                    # Check excludes
                    if excludes and any(fnmatch(str_path, exc) for exc in excludes):
                        print(f"In excludes list")
                        continue

                    # Get file information
                    try:
                        stat = item_path.stat()
                        matches.append({
                            "path": str_path,
                            "type": "file" if item_path.is_file() else "directory",
                            "size": stat.st_size if item_path.is_file() else None,
                            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
                        })
                        print(f"Added to matches")
                    except (OSError, PermissionError):
                        print(f"Failed to get file info")
                        continue

            return {
                "matches": matches,
                "total": len(matches)
            }

        except Exception as e:
            print(f"Error during search: {str(e)}")
            return {
                "error": f"Search failed: {str(e)}",
                "matches": []
            }


async def main():
    """Test the FindByNameTool with various scenarios."""
    tool = FindByNameTool()

    # Test case 1: Find all Python files
    print("\n1. Finding all Python files in current directory:")
    result = await tool.execute(
        search_directory=os.path.dirname(os.path.dirname(os.path.dirname(__file__))),  # Go to project root
        pattern="*.py",
        type="file"
    )
    print(json.dumps(result, indent=2))

    # Test case 2: Find files with depth limit
    print("\n2. Finding files with depth limit:")
    result = await tool.execute(
        search_directory=".",
        pattern="*.*",
        max_depth=1,
        type="file"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    asyncio.run(main())

from pathlib import Path
from typing import Any, Dict

from .base import BaseTool


class ListDirTool(BaseTool):
    """Tool for listing directory contents with detailed information."""

    @property
    def name(self) -> str:
        return "list_dir"

    @property
    def description(self) -> str:
        return """List the contents of a directory. For each child in the directory,
        output will have: relative path, whether it is a directory or file,
        size in bytes if file, and number of children if directory."""

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "directory_path": {
                    "type": "string",
                    "description": "Path to list contents of"
                }
            },
            "required": ["directory_path"]
        }

    def execute(self, directory_path: str) -> Dict[str, Any]:
        """List directory contents with detailed information.
        
        Args:
            directory_path: Path to the directory to list
            
        Returns:
            Dict containing directory contents with details
        """
        try:
            path = Path(directory_path)
            if not path.is_dir():
                return {
                    "error": f"Path {directory_path} is not a directory",
                    "contents": []
                }

            contents = []
            for item in path.iterdir():
                item_info = {
                    "name": item.name,
                    "path": str(item.relative_to(path)),
                    "is_dir": item.is_dir()
                }

                if item.is_file():
                    try:
                        item_info["size"] = item.stat().st_size
                        item_info["last_modified"] = item.stat().st_mtime
                    except OSError:
                        item_info["size"] = 0
                        item_info["last_modified"] = 0
                elif item.is_dir():
                    try:
                        # Count number of items in directory (non-recursive)
                        item_info["child_count"] = sum(1 for _ in item.iterdir())
                    except (OSError, PermissionError):
                        item_info["child_count"] = 0

                contents.append(item_info)

            # Sort contents: directories first, then files, both alphabetically
            contents.sort(key=lambda x: (not x["is_dir"], x["name"].lower()))

            return {
                "directory": str(path),
                "contents": contents
            }

        except PermissionError:
            return {
                "error": f"Permission denied accessing {directory_path}",
                "contents": []
            }
        except Exception as e:
            return {
                "error": f"Error listing directory {directory_path}: {str(e)}",
                "contents": []
            }


if __name__ == '__main__':
    pass

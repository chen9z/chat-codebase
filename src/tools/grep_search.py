import subprocess
from typing import Any, Dict, List, Optional

from src.tools.base import BaseTool


class GrepSearchTool(BaseTool):
    """Tool for performing fast text-based search using ripgrep."""

    @property
    def name(self) -> str:
        return "grep_search"

    @property
    def description(self) -> str:
        return """Fast text-based search that finds exact pattern matches within files or directories,
        utilizing the ripgrep command for efficient searching."""

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "search_directory": {
                    "type": "string",
                    "description": "The directory from which to run the ripgrep command."
                },
                "query": {
                    "type": "string",
                    "description": "The search term or pattern to look for within files."
                },
                "match_per_line": {
                    "type": "boolean",
                    "description": "If true, returns each line that matches the query with line numbers."
                },
                "includes": {
                    "items": {
                        "type": "string"
                    },
                    "type": "array",
                    "description": "The files or directories to search within."
                },
                "case_insensitive": {
                    "type": "boolean",
                    "description": "If true, performs a case-insensitive search."
                }
            },
            "required": ["query"]
        }

    def execute(
            self,
            query: str,
            search_directory: str = ".",
            match_per_line: bool = True,
            includes: Optional[List[str]] = None,
            case_insensitive: bool = False
    ) -> Dict[str, Any]:
        """Execute grep search using ripgrep.
        
        Args:
            query: Search pattern
            search_directory: Directory to search in
            match_per_line: Whether to show line numbers and content
            includes: File patterns to include
            case_insensitive: Whether to ignore case
            
        Returns:
            Dict containing search results
        """
        cmd = ["rg"]

        # Add options
        if case_insensitive:
            cmd.append("-i")

        if match_per_line:
            cmd.extend(["-n", "--color", "never"])
        else:
            cmd.append("-l")

        # Add file type includes
        if includes:
            for include in includes:
                cmd.extend(["-g", include])

        # Add search pattern and directory
        cmd.extend([query, search_directory])

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False  # Don't raise on non-zero exit (no matches)
            )

            matches = []
            if result.stdout:
                lines = result.stdout.strip().split("\n")
                for line in lines[:50]:  # Limit to 50 matches
                    if match_per_line:
                        # Parse output format: file:line:content
                        parts = line.split(":", 2)
                        if len(parts) >= 3:
                            matches.append({
                                "file": parts[0],
                                "line": int(parts[1]),
                                "content": parts[2]
                            })
                    else:
                        matches.append({"file": line})

            return {
                "matches": matches,
                "truncated": len(matches) >= 50
            }

        except subprocess.CalledProcessError as e:
            return {
                "error": f"Search failed: {str(e)}",
                "matches": []
            }
        except Exception as e:
            return {
                "error": f"Unexpected error: {str(e)}",
                "matches": []
            }


if __name__ == '__main__':
    tool = GrepSearchTool()

    # Test case 1: Search for "class" in the current directory
    print("\n1. Searching for 'class' in current directory:")
    result = tool.execute(query="class", search_directory=".")
    print(result)

    # Test case 2: Search for "def" in the src directory
    print("\n2. Searching for 'def' in the src directory:")
    result = tool.execute(query="def", search_directory="src")
    print(result)

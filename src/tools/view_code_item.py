from typing import Any, Dict, List, Optional, Tuple
import ast
from pathlib import Path

from .base import BaseTool


class ViewCodeItemTool(BaseTool):
    """Tool for viewing specific code items like functions and classes."""

    @property
    def name(self) -> str:
        return "view_code_item"

    @property
    def description(self) -> str:
        return """View the content of a code item node, such as a class or a function in a file.
        You must use a fully qualified code item name."""

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the file to find the code node"
                },
                "node_name": {
                    "type": "string",
                    "description": "The name of the node to view"
                }
            },
            "required": ["file_path", "node_name"]
        }

    def execute(self, file_path: str, node_name: str) -> Dict[str, Any]:
        """View a specific code item from a file.
        
        Args:
            file_path: Path to the file containing the code item
            node_name: Name of the code item to view (e.g., "ClassName.method_name")
            
        Returns:
            Dict containing the code item contents and metadata
        """
        try:
            path = Path(file_path)
            if not path.is_file():
                return {
                    "error": f"Path {file_path} is not a file",
                    "contents": None
                }

            with path.open('r', encoding='utf-8') as f:
                source = f.read()
                tree = ast.parse(source)

                # Split node name into parts (for nested items)
                name_parts = node_name.split('.')

                # Find the node and its location
                node_info = self._find_node(tree, name_parts)
                if not node_info:
                    return {
                        "error": f"Code item '{node_name}' not found in {file_path}",
                        "contents": None
                    }

                node, start_line, end_line = node_info

                # Get the source lines for the node
                source_lines = source.splitlines()
                node_source = source_lines[start_line - 1:end_line]

                return {
                    "file": str(path),
                    "node_name": node_name,
                    "node_type": type(node).__name__,
                    "start_line": start_line,
                    "end_line": end_line,
                    "contents": node_source
                }

        except SyntaxError as e:
            return {
                "error": f"Syntax error in file {file_path}: {str(e)}",
                "contents": None
            }
        except Exception as e:
            return {
                "error": f"Error reading code item from {file_path}: {str(e)}",
                "contents": None
            }

    def _find_node(self, tree: ast.AST, name_parts: List[str]) -> Optional[Tuple[ast.AST, int, int]]:
        """Find a node in the AST by its name parts.
        
        Args:
            tree: The AST to search
            name_parts: List of name parts (e.g., ["ClassName", "method_name"])
            
        Returns:
            Tuple of (node, start_line, end_line) if found, None otherwise
        """

        def find_in_node(node: ast.AST, remaining_parts: List[str]) -> Optional[Tuple[ast.AST, int, int]]:
            if not remaining_parts:
                return None

            current_name = remaining_parts[0]

            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)) and \
                        child.name == current_name:
                    if len(remaining_parts) == 1:
                        return child, child.lineno, child.end_lineno or child.lineno
                    return find_in_node(child, remaining_parts[1:])

            return None

        return find_in_node(tree, name_parts)

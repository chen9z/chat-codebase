import asyncio
import json
from pathlib import Path
from typing import Any, Dict

from src.tools.base import BaseTool  # Try relative import first


class RelatedFilesTool(BaseTool):
    """Tool for finding related files to a given input file."""

    @property
    def name(self) -> str:
        return "related_files"

    @property
    def description(self) -> str:
        return """Finds other files that are related to or commonly used with the input file.
        Useful for retrieving adjacent files to understand context or make next edits."""

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Input file path"
                }
            },
            "required": ["file_path"]
        }

    async def execute(self, file_path: str) -> Dict[str, Any]:
        """Find related files for the given input file.
        
        Args:
            file_path: Path to the input file
            
        Returns:
            Dict containing related files grouped by relationship type
        """
        try:
            path = Path(file_path)
            if not path.is_file():
                return {
                    "error": f"Path {file_path} is not a file",
                    "related_files": {}
                }

            related = {
                "same_directory": [],
                "test_files": [],
                "implementation_files": [],
                "documentation": [],
                "configuration": []
            }

            # Get the file's directory and name parts
            directory = path.parent
            name_stem = path.stem
            name_suffix = path.suffix

            # Find files in the same directory
            for item in directory.iterdir():
                if item == path:  # Skip the input file itself
                    continue

                if item.is_file():
                    item_stem = item.stem
                    item_suffix = item.suffix
                    rel_path = str(item.relative_to(directory.parent))

                    # Test files
                    if (item_stem.startswith('test_') and item_stem[5:] == name_stem) or \
                            (item_stem.endswith('_test') and item_stem[:-5] == name_stem) or \
                            (item_stem == name_stem + '_test'):
                        related["test_files"].append(rel_path)
                        continue

                    # Implementation files
                    if name_suffix in ['.h', '.hpp'] and item_suffix in ['.c', '.cpp'] and item_stem == name_stem:
                        related["implementation_files"].append(rel_path)
                        continue

                    if name_suffix == '.ts' and item_suffix == '.js' and item_stem == name_stem:
                        related["implementation_files"].append(rel_path)
                        continue

                    # Documentation
                    if item_stem == name_stem and item_suffix in ['.md', '.rst', '.txt']:
                        related["documentation"].append(rel_path)
                        continue

                    # Configuration files
                    if item_suffix in ['.json', '.yaml', '.yml', '.toml', '.ini', '.config']:
                        related["configuration"].append(rel_path)
                        continue

                    # Other files in same directory
                    related["same_directory"].append(rel_path)

            # Remove empty categories
            related = {k: v for k, v in related.items() if v}

            return {
                "file": str(path),
                "related_files": related
            }

        except Exception as e:
            return {
                "error": f"Failed to find related files: {str(e)}",
                "related_files": {}
            }


async def main():
    """Test the RelatedFilesTool with various scenarios."""
    tool = RelatedFilesTool()

    # Test case 1: Find related files for a Python file
    print("\n1. Finding related files for a Python file:")
    result = await tool.execute(
        file_path="base.py"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    asyncio.run(main())

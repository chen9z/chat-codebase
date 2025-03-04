import tree_sitter_css
import tree_sitter_go
import tree_sitter_html
import tree_sitter_java
import tree_sitter_javascript
import tree_sitter_python
import tree_sitter_typescript
import tree_sitter_xml
import tree_sitter_yaml

from src.data.splitter import Splitter, Document
from tree_sitter import Language


class CodeSpliter(Splitter):

    def __init__(self, chunk_size: int, chunk_overlap: int):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.suffix_to_lang = {
            "java": Language(tree_sitter_java.language()),
            "py": Language(tree_sitter_python.language()),
            "go": Language(tree_sitter_go.language()),
            "js": Language(tree_sitter_javascript.language()),
            "ts": Language(tree_sitter_typescript.language()),
            "css": Language(tree_sitter_css.language()),
            "html": Language(tree_sitter_html.language()),
            "xml": Language(tree_sitter_xml.language()),
            "yml": Language(tree_sitter_yaml.language()),
            "yaml": Language(tree_sitter_yaml.language()),
        }

    def split(self, file_path: str) -> list[Document]:
        pass

    def split_text(self, text: str) -> list[Document]:
        pass


if __name__ == '__main__':
    print()

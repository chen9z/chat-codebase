from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import tree_sitter
import config.settings as settings
from tree_sitter_languages import get_parser

from src.tools.base import BaseTool


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

    def _get_parser_for_file(self, file_path):
        lang = settings.ext_to_lang.get(Path(file_path).suffix)
        return get_parser(lang)

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
                source_code = f.read()
                source_lines = source_code.splitlines()

                # Get the appropriate parser for this file type
                parser = self._get_parser_for_file(file_path)

                # Parse the code
                tree = parser.parse(bytes(source_code, 'utf-8'))

                # Split node name into parts (for nested items)
                name_parts = node_name.split('.')

                # Find the node
                node_info = self._find_node(tree.root_node, name_parts, source_code)

                if not node_info:
                    return {
                        "error": f"Code item '{node_name}' not found in {file_path}",
                        "contents": None
                    }

                node, start_line, end_line = node_info

                # Get the source lines for the node
                node_source = source_lines[start_line - 1:end_line]

                return {
                    "file": str(path),
                    "node_name": node_name,
                    "node_type": node.type,
                    "start_line": start_line,
                    "end_line": end_line,
                    "contents": node_source
                }

        except Exception as e:
            return {
                "error": f"Error reading code item from {file_path}: {str(e)}",
                "contents": None
            }

    def _find_node(self, root_node, name_parts, source_code) -> Optional[Tuple[tree_sitter.Node, int, int]]:
        """Find a node in the tree-sitter tree by its name parts across multiple languages."""
        if not name_parts:
            return None
        
        # Debug the tree structure to identify language
        node_types = set(child.type for child in root_node.children)
        
        # Better language detection based on file content and extension
        is_python = 'class_definition' in node_types or 'function_definition' in node_types
        
        # TypeScript specific markers
        has_ts_markers = 'lexical_declaration' in node_types or 'type_annotation' in node_types
        
        # JavaScript/TypeScript detection
        is_js_ts = ('class_declaration' in node_types or 
                    'function_declaration' in node_types or 
                    'lexical_declaration' in node_types)
        
        # Java has method_declaration but no lexical_declaration
        # Java would have specific patterns like public/private modifiers
        is_java = (('method_declaration' in node_types or 
                   ('class_declaration' in node_types and 'interface_declaration' in node_types))
                  and not has_ts_markers)
               
        # If we have TypeScript markers, prioritize TypeScript over Java
        if has_ts_markers and is_java:
            is_java = False
        
        # Print debug info about the detected language and node types
        print(f"Detected node types: {node_types}")
        print(f"Language detection: Python={is_python}, JS/TS={is_js_ts}, Java={is_java}")
        
        # Define node types based on language
        if is_java:
            class_types = ('class_declaration', 'interface_declaration')
            function_types = ('method_declaration', 'constructor_declaration')
            block_types = ('class_body', 'block')
        elif is_js_ts:
            class_types = ('class_declaration', 'interface_declaration')
            function_types = ('function_declaration', 'method_definition', 'function', 'arrow_function', 'method')
            block_types = ('class_body', 'statement_block')
            variable_types = ('lexical_declaration', 'variable_declaration')
        else:  # Python (default)
            class_types = ('class_definition',)
            function_types = ('function_definition',)
            block_types = ('block',)
        
        # Helper function to print tree structure for debugging
        def print_tree(node, indent=0, max_depth=3):
            if indent > max_depth:
                return
            
            print(f"{'  ' * indent}[{node.type}] {source_code[node.start_byte:node.end_byte][:30]}...")
            for child in node.children:
                print_tree(child, indent + 1, max_depth)
        
        # Helper function to get node name based on language
        def get_node_name(node):
            # JS/TS class methods and declarations
            if node.type == 'method_definition':
                for child in node.children:
                    if child.type == 'property_identifier':
                        return source_code[child.start_byte:child.end_byte]
            
            # JS/TS function declarations
            elif node.type == 'function_declaration':
                for child in node.children:
                    if child.type == 'identifier':
                        return source_code[child.start_byte:child.end_byte]
            
            # Java methods
            elif node.type == 'method_declaration':
                for child in node.children:
                    if child.type == 'identifier':
                        return source_code[child.start_byte:child.end_byte]
            
            # Java class declarations
            elif node.type == 'class_declaration':
                for child in node.children:
                    if child.type == 'identifier':
                        return source_code[child.start_byte:child.end_byte]
            
            # Variable declarations (for arrow functions in TS)
            elif node.type in ('lexical_declaration', 'variable_declaration'):
                for child in node.children:
                    if child.type == 'variable_declarator':
                        for subchild in child.children:
                            if subchild.type == 'identifier':
                                return source_code[subchild.start_byte:subchild.end_byte]
            
            # Python and general case: Find the identifier
            for child in node.children:
                if child.type == 'identifier' or child.type == 'property_identifier':
                    return source_code[child.start_byte:child.end_byte]
                
            return None
        
        # Helper function to find class body more reliably across languages
        def find_body(node):
            # First try direct class_body for JS/TS/Java
            for child in node.children:
                if child.type == 'class_body':
                    return child
            
            # For Python or fallback
            for child in node.children:
                if child.type == 'block':
                    return child
            
            return None
        
        # Search for single name (standalone function or method)
        if len(name_parts) == 1:
            target_name = name_parts[0]
            
            # First try to find it as a top-level function
            for child in root_node.children:
                # Handle Python and JS function declarations
                if child.type in function_types:
                    node_name = get_node_name(child)
                    if node_name == target_name:
                        start_line = child.start_point[0] + 1
                        end_line = child.end_point[0] + 1
                        return child, start_line, end_line
                
                # Handle TypeScript/JavaScript variable declarations (for arrow functions)
                if is_js_ts and child.type in ('lexical_declaration', 'variable_declaration'):
                    for decl in child.children:
                        if decl.type == 'variable_declarator':
                            for id_node in decl.children:
                                if id_node.type == 'identifier':
                                    var_name = source_code[id_node.start_byte:id_node.end_byte]
                                    if var_name == target_name:
                                        start_line = child.start_point[0] + 1
                                        end_line = child.end_point[0] + 1
                                        return child, start_line, end_line
            
            # If not found as standalone, search in all classes
            for child in root_node.children:
                if child.type in class_types:
                    class_body = find_body(child)
                    if class_body:
                        # Search in class body for the method
                        for method in class_body.children:
                            if method.type in function_types:
                                node_name = get_node_name(method)
                                if node_name == target_name:
                                    start_line = method.start_point[0] + 1
                                    end_line = method.end_point[0] + 1
                                    return method, start_line, end_line
            
            # Not found
            print(f"Could not find standalone function or method named: {target_name}")
            return None
        
        # Handle fully qualified names (Class.method or Class.InnerClass.method)
        class_name = name_parts[0]
        
        # For TypeScript/JavaScript - we need to examine each class declaration more carefully
        if is_js_ts and not is_java:
            # Find the class first
            for node in root_node.children:
                if node.type == 'class_declaration':
                    # Check each child to find the name (can be in different positions)
                    found_name = None
                    for child in node.children:
                        if child.type == 'identifier':
                            found_name = source_code[child.start_byte:child.end_byte]
                        elif child.type == 'type_identifier':
                            found_name = source_code[child.start_byte:child.end_byte]
                    
                    if found_name == class_name:
                        # Found the class, now get its body
                        body_node = None
                        for child in node.children:
                            if child.type == 'class_body':
                                body_node = child
                                break
                        
                        if body_node and len(name_parts) == 2:
                            # Look for the method
                            method_name = name_parts[1]
                            for method in body_node.children:
                                # Check for TypeScript methods (method_definition)
                                if method.type == 'method_definition':
                                    for prop in method.children:
                                        if prop.type == 'property_identifier':
                                            if source_code[prop.start_byte:prop.end_byte] == method_name:
                                                start_line = method.start_point[0] + 1
                                                end_line = method.end_point[0] + 1
                                                return method, start_line, end_line
                                            
                                # Also check for property with method value
                                elif method.type == 'public_field_definition':
                                    prop_name = None
                                    for prop in method.children:
                                        if prop.type == 'property_identifier':
                                            prop_name = source_code[prop.start_byte:prop.end_byte]
                                        
                                    if prop_name == method_name:
                                        start_line = method.start_point[0] + 1
                                        end_line = method.end_point[0] + 1
                                        return method, start_line, end_line
                        
                        break
        
        # For Java - handle Java-specific class and method structures 
        elif is_java:
            # Find the class declaration node
            class_node = None
            
            # Debug Java AST structure
            for node in root_node.children:
                if node.type in ('class_declaration', 'interface_declaration'):
                    print(f"Found Java {node.type}")
                    # Print more helpful structure information
                    for child in node.children:
                        print(f"  Child node: {child.type}")
                        if child.type == 'identifier':
                            print(f"    Name: {source_code[child.start_byte:child.end_byte]}")
                        elif child.type == 'class_body':
                            print(f"    Found class body with {len(child.children)} children")
                            for body_child in child.children:
                                if body_child.type == 'method_declaration':
                                    method_name = None
                                    for mc in body_child.children:
                                        if mc.type == 'identifier':
                                            method_name = source_code[mc.start_byte:mc.end_byte]
                                            break
                                    print(f"      Method: {method_name} ({body_child.type})")
                                elif body_child.type == 'class_declaration':  # inner class
                                    inner_name = None
                                    for ic in body_child.children:
                                        if ic.type == 'identifier':
                                            inner_name = source_code[ic.start_byte:ic.end_byte]
                                            break
                                    print(f"      Inner class: {inner_name}")
            
            # Find target class based on first name part
            for node in root_node.children:
                if node.type in ('class_declaration', 'interface_declaration'):
                    for child in node.children:
                        if child.type == 'identifier':
                            node_name = source_code[child.start_byte:child.end_byte]
                            if node_name == class_name:
                                class_node = node
                                break
                    if class_node:
                        break
            
            if class_node:
                # Find the class body
                class_body = None
                for child in class_node.children:
                    if child.type == 'class_body':
                        class_body = child
                        break
                
                if class_body:
                    if len(name_parts) == 2:  # Class.method
                        method_name = name_parts[1]
                        # Directly search for methods in the class body
                        for node in class_body.children:
                            if node.type == 'method_declaration':
                                # Find the method name
                                for child in node.children:
                                    if child.type == 'identifier':
                                        node_name = source_code[child.start_byte:child.end_byte]
                                        if node_name == method_name:
                                            start_line = node.start_point[0] + 1
                                            end_line = node.end_point[0] + 1
                                            return node, start_line, end_line
                    
                    elif len(name_parts) == 3:  # Class.InnerClass.method
                        inner_class_name = name_parts[1]
                        method_name = name_parts[2]
                        
                        # Find inner class
                        inner_class = None
                        for node in class_body.children:
                            if node.type == 'class_declaration':
                                for child in node.children:
                                    if child.type == 'identifier':
                                        node_name = source_code[child.start_byte:child.end_byte]
                                        if node_name == inner_class_name:
                                            inner_class = node
                                            break
                                if inner_class:
                                    break
                        
                        if inner_class:
                            # Find the inner class body
                            inner_body = None
                            for child in inner_class.children:
                                if child.type == 'class_body':
                                    inner_body = child
                                    break
                            
                            if inner_body:
                                # Find the method in inner class
                                for node in inner_body.children:
                                    if node.type == 'method_declaration':
                                        for child in node.children:
                                            if child.type == 'identifier':
                                                node_name = source_code[child.start_byte:child.end_byte]
                                                if node_name == method_name:
                                                    start_line = node.start_point[0] + 1
                                                    end_line = node.end_point[0] + 1
                                                    return node, start_line, end_line
        
        # Python case (as fallback)
        else:
            # Find the class first
            class_node = None
            for node in root_node.children:
                if node.type in class_types:
                    node_name = get_node_name(node)
                    if node_name == class_name:
                        class_node = node
                        break
            
            if class_node:
                # Find the class body
                body_node = find_body(class_node)
                if body_node:
                    # Handle nested classes if more than 2 parts (e.g., Class.NestedClass.method)
                    current_node = body_node
                    for i in range(1, len(name_parts) - 1):
                        nested_class_name = name_parts[i]
                        found = False
                        
                        for child in current_node.children:
                            if child.type in class_types:
                                node_name = get_node_name(child)
                                if node_name == nested_class_name:
                                    nested_body = find_body(child)
                                    if nested_body:
                                        current_node = nested_body
                                        found = True
                                        break
                        
                        if not found:
                            print(f"Could not find nested class: {nested_class_name}")
                            return None
                    
                    # Find the method in the final class body
                    method_name = name_parts[-1]
                    for child in current_node.children:
                        if child.type == 'function_definition':
                            node_name = get_node_name(child)
                            if node_name == method_name:
                                start_line = child.start_point[0] + 1
                                end_line = child.end_point[0] + 1
                                return child, start_line, end_line
        
        # Use more specific error messages for debugging
        if len(name_parts) > 1:
            print(f"Could not find method {name_parts[-1]} in class {name_parts[0]}")
        
        return None

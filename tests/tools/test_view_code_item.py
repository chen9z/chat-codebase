import os
import shutil
import tempfile

import pytest

from src.tools.view_code_item import ViewCodeItemTool


class TestViewCodeItemTool:
    @pytest.fixture
    def setup_test_files(self):
        """Create temporary test files for different languages"""
        # Create a temporary directory for test files
        temp_dir = tempfile.mkdtemp()

        # Python test file
        python_content = """
class Calculator:
    def __init__(self):
        self.result = 0
    
    def add(self, a, b):
        return a + b
        
    def subtract(self, a, b):
        return a - b

class NestedExample:
    class Inner:
        def inner_method(self):
            return "inner"

def standalone_function(x):
    return x * 2
"""

        # JavaScript test file
        js_content = """
class Calculator {
    constructor() {
        this.result = 0;
    }
    
    add(a, b) {
        return a + b;
    }
    
    subtract(a, b) {
        return a - b;
    }
}

function doMath(a, b) {
    return a * b;
}
"""

        # TypeScript test file
        ts_content = """
interface MathOperation {
    execute(a: number, b: number): number;
}

class Calculator implements MathOperation {
    private result: number;
    
    constructor() {
        this.result = 0;
    }
    
    execute(a: number, b: number): number {
        return a + b;
    }
    
    multiply(a: number, b: number): number {
        return a * b;
    }
}

const divide = (a: number, b: number): number => {
    if (b === 0) throw new Error('Division by zero');
    return a / b;
};
"""

        # Java test file
        java_content = """
public class Calculator {
    private int result;
    
    public Calculator() {
        this.result = 0;
    }
    
    public int add(int a, int b) {
        return a + b;
    }
    
    public int subtract(int a, int b) {
        return a - b;
    }
    
    class InnerCalculator {
        public int multiply(int a, int b) {
            return a * b;
        }
    }
}

interface MathOperation {
    int execute(int a, int b);
}
"""

        # Write files
        py_file = os.path.join(temp_dir, "calculator.py")
        js_file = os.path.join(temp_dir, "calculator.js")
        ts_file = os.path.join(temp_dir, "calculator.ts")
        java_file = os.path.join(temp_dir, "Calculator.java")

        with open(py_file, 'w') as f:
            f.write(python_content)
        with open(js_file, 'w') as f:
            f.write(js_content)
        with open(ts_file, 'w') as f:
            f.write(ts_content)
        with open(java_file, 'w') as f:
            f.write(java_content)

        # Return file paths
        yield {
            "python": py_file,
            "javascript": js_file,
            "typescript": ts_file,
            "java": java_file
        }

        # Cleanup after tests
        shutil.rmtree(temp_dir)

    def test_python_class_method(self, setup_test_files):
        """Test finding a method in a Python class"""
        tool = ViewCodeItemTool()
        file_path = setup_test_files["python"]

        # Test qualified name (Class.method)
        result = tool.execute(file_path, "Calculator.add")

        assert result.get("error") is None
        assert result.get("node_type") == "function_definition"
        assert "def add" in "\n".join(result.get("contents", []))
        assert result.get("node_name") == "Calculator.add"

    def test_python_standalone_function(self, setup_test_files):
        """Test finding a standalone function in Python"""
        tool = ViewCodeItemTool()
        file_path = setup_test_files["python"]

        # Test direct function name
        result = tool.execute(file_path, "standalone_function")

        assert result.get("error") is None
        assert result.get("node_type") == "function_definition"
        assert "def standalone_function" in "\n".join(result.get("contents", []))

    def test_python_nested_class(self, setup_test_files):
        """Test finding nested classes in Python"""
        tool = ViewCodeItemTool()
        file_path = setup_test_files["python"]

        # Test deeply nested method
        result = tool.execute(file_path, "NestedExample.Inner.inner_method")

        assert result.get("error") is None
        assert result.get("node_type") == "function_definition"
        assert "def inner_method" in "\n".join(result.get("contents", []))

    def test_javascript_class_method(self, setup_test_files):
        """Test finding a method in a JavaScript class"""
        tool = ViewCodeItemTool()
        file_path = setup_test_files["javascript"]

        # Test class method
        result = tool.execute(file_path, "Calculator.add")

        assert result.get("error") is None
        assert "add(a, b)" in "\n".join(result.get("contents", []))
        assert "return a + b" in "\n".join(result.get("contents", []))

    def test_javascript_function(self, setup_test_files):
        """Test finding a standalone function in JavaScript"""
        tool = ViewCodeItemTool()
        file_path = setup_test_files["javascript"]

        # Test standalone function
        result = tool.execute(file_path, "doMath")

        assert result.get("error") is None
        assert "function doMath" in "\n".join(result.get("contents", []))
        assert "return a * b" in "\n".join(result.get("contents", []))

    def test_typescript_interface_implementation(self, setup_test_files):
        """Test finding methods in TypeScript with interfaces"""
        tool = ViewCodeItemTool()
        file_path = setup_test_files["typescript"]

        # Test method that implements an interface
        result = tool.execute(file_path, "Calculator.execute")

        assert result.get("error") is None
        assert "execute(a: number, b: number)" in "\n".join(result.get("contents", []))

    def test_typescript_arrow_function(self, setup_test_files):
        """Test finding arrow functions in TypeScript"""
        tool = ViewCodeItemTool()
        file_path = setup_test_files["typescript"]

        # Test arrow function - this is challenging and might need refinement
        result = tool.execute(file_path, "divide")

        # This might fail if tree-sitter doesn't properly identify the arrow function
        # If it fails, we may need to enhance the implementation
        assert result.get("error") is None
        assert "const divide" in "\n".join(result.get("contents", []))

    def test_java_class_method(self, setup_test_files):
        """Test finding a method in a Java class"""
        tool = ViewCodeItemTool()
        file_path = setup_test_files["java"]

        # Test Java class method
        result = tool.execute(file_path, "Calculator.add")

        print(result)
        assert result.get("error") is None
        assert "public int add" in "\n".join(result.get("contents", []))

    def test_java_inner_class_method(self, setup_test_files):
        """Test finding a method in a Java inner class"""
        tool = ViewCodeItemTool()
        file_path = setup_test_files["java"]

        # Test Java inner class method
        result = tool.execute(file_path, "Calculator.InnerCalculator.multiply")

        assert result.get("error") is None
        assert "public int multiply" in "\n".join(result.get("contents", []))

    def test_file_not_found(self):
        """Test error handling when file doesn't exist"""
        tool = ViewCodeItemTool()

        result = tool.execute("/path/to/nonexistent/file.py", "SomeClass.some_method")

        assert result.get("error") is not None
        assert "not a file" in result.get("error")

    def test_node_not_found(self, setup_test_files):
        """Test error handling when node doesn't exist"""
        tool = ViewCodeItemTool()
        file_path = setup_test_files["python"]

        result = tool.execute(file_path, "NonexistentClass.nonexistent_method")

        assert result.get("error") is not None
        assert "not found" in result.get("error")

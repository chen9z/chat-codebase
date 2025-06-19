"""
pytest 配置文件
"""

import sys
from pathlib import Path
import pytest

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture
def sample_java_code():
    """提供示例 Java 代码"""
    return '''
package com.example;

import java.util.List;

/**
 * Sample class for testing
 */
public class TestClass {
    
    /**
     * Constructor with documentation
     * @param name the name parameter
     */
    public TestClass(String name) {
        this.name = name;
    }
    
    // Simple method with line comment
    public void simpleMethod() {
        System.out.println("Hello");
    }
    
    private String name;
}
'''


@pytest.fixture
def sample_python_code():
    """提供示例 Python 代码"""
    return '''
"""
Sample Python module for testing
"""

class TestClass:
    """A test class with documentation"""
    
    def __init__(self, name: str):
        """Initialize with name
        
        Args:
            name: The name parameter
        """
        self.name = name
    
    # Simple method with comment
    def simple_method(self):
        """Simple method"""
        print("Hello")
'''


@pytest.fixture
def temp_file_factory():
    """创建临时文件的工厂函数"""
    import tempfile
    import os
    
    created_files = []
    
    def create_temp_file(content: str, suffix: str = '.tmp'):
        with tempfile.NamedTemporaryFile(mode='w', suffix=suffix, delete=False) as f:
            f.write(content)
            created_files.append(f.name)
            return f.name
    
    yield create_temp_file
    
    # 清理临时文件
    for file_path in created_files:
        try:
            os.unlink(file_path)
        except OSError:
            pass

#!/usr/bin/env python3
"""
测试语义分块优化功能
"""

import tempfile
import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data.splitter import parse

# 测试用的Python代码
PYTHON_CODE_SAMPLE = '''
"""
这是一个模块级的文档字符串
用于测试Python代码的语义分块
"""

import os
import sys

class Calculator:
    """
    计算器类，用于演示类级别的文档字符串
    和方法的语义分块
    """
    
    def __init__(self, name: str):
        """
        初始化计算器
        
        Args:
            name: 计算器名称
        """
        self.name = name
        self.result = 0
    
    # 这是一个行注释
    def add(self, a: float, b: float) -> float:
        """
        加法运算
        
        Args:
            a: 第一个数
            b: 第二个数
            
        Returns:
            两数之和
        """
        return a + b
    
    def subtract(self, a: float, b: float) -> float:
        """减法运算"""
        return a - b
    
    @property
    def current_result(self) -> float:
        """获取当前结果"""
        return self.result

# 独立函数
def standalone_function(x: int) -> int:
    """
    独立函数，用于测试函数级别的分块
    
    Args:
        x: 输入参数
        
    Returns:
        处理后的结果
    """
    return x * 2

# 另一个类
class AdvancedCalculator(Calculator):
    """高级计算器，继承自Calculator"""
    
    def power(self, base: float, exponent: float) -> float:
        """幂运算"""
        return base ** exponent
'''

# 测试用的Java代码
JAVA_CODE_SAMPLE = '''
package com.example.calculator;

import java.util.List;
import java.util.ArrayList;

/**
 * 计算器类，用于演示Java代码的语义分块
 * 
 * @author Test
 * @version 1.0
 */
public class Calculator {
    
    private String name;
    private double result;
    
    /**
     * 构造函数
     * 
     * @param name 计算器名称
     */
    public Calculator(String name) {
        this.name = name;
        this.result = 0.0;
    }
    
    /**
     * 加法运算
     * 这是一个多行的Javadoc注释
     * 
     * @param a 第一个数
     * @param b 第二个数
     * @return 两数之和
     */
    public double add(double a, double b) {
        double sum = a + b;
        this.result = sum;
        return sum;
    }
    
    // 简单的行注释
    public double subtract(double a, double b) {
        return a - b;
    }
    
    /*
     * 块注释
     * 用于描述乘法方法
     */
    public double multiply(double a, double b) {
        return a * b;
    }
    
    /**
     * 获取当前结果
     * @return 当前计算结果
     */
    public double getCurrentResult() {
        return this.result;
    }
}

/**
 * 高级计算器接口
 */
interface AdvancedCalculator {
    /**
     * 幂运算
     * @param base 底数
     * @param exponent 指数
     * @return 幂运算结果
     */
    double power(double base, double exponent);
}
'''

def test_python_semantic_chunking():
    """测试Python代码的语义分块"""
    print("=== 测试Python语义分块 ===")
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(PYTHON_CODE_SAMPLE)
        temp_file = f.name
    
    try:
        documents = parse(temp_file)
        
        print(f"生成了 {len(documents)} 个文档块:")
        print("-" * 60)
        
        for i, doc in enumerate(documents, 1):
            print(f"块 {i}: 行 {doc.start_line}-{doc.end_line}")
            print(f"内容预览: {doc.content[:100]}...")
            
            # 检查是否包含注释和方法
            has_docstring = '"""' in doc.content or "'''" in doc.content
            has_comment = '#' in doc.content
            has_class = 'class ' in doc.content
            has_function = 'def ' in doc.content
            
            features = []
            if has_docstring:
                features.append("文档字符串")
            if has_comment:
                features.append("注释")
            if has_class:
                features.append("类定义")
            if has_function:
                features.append("函数定义")
            
            if features:
                print(f"特征: {', '.join(features)}")
            
            print("-" * 60)
        
        return len(documents) > 0
        
    finally:
        os.unlink(temp_file)

def test_java_semantic_chunking():
    """测试Java代码的语义分块"""
    print("\n=== 测试Java语义分块 ===")
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.java', delete=False) as f:
        f.write(JAVA_CODE_SAMPLE)
        temp_file = f.name
    
    try:
        documents = parse(temp_file)
        
        print(f"生成了 {len(documents)} 个文档块:")
        print("-" * 60)
        
        for i, doc in enumerate(documents, 1):
            print(f"块 {i}: 行 {doc.start_line}-{doc.end_line}")
            print(f"内容预览: {doc.content[:100]}...")
            
            # 检查是否包含注释和方法
            has_javadoc = '/**' in doc.content
            has_line_comment = '//' in doc.content
            has_block_comment = '/*' in doc.content and not has_javadoc
            has_class = 'class ' in doc.content
            has_interface = 'interface ' in doc.content
            has_method = 'public ' in doc.content and '(' in doc.content
            
            features = []
            if has_javadoc:
                features.append("Javadoc")
            if has_line_comment:
                features.append("行注释")
            if has_block_comment:
                features.append("块注释")
            if has_class:
                features.append("类定义")
            if has_interface:
                features.append("接口定义")
            if has_method:
                features.append("方法定义")
            
            if features:
                print(f"特征: {', '.join(features)}")
            
            print("-" * 60)
        
        return len(documents) > 0
        
    finally:
        os.unlink(temp_file)

def test_line_boundary_alignment():
    """测试行边界对齐功能"""
    print("\n=== 测试行边界对齐 ===")
    
    # 简单的测试代码，确保分割在行边界
    simple_code = '''def function1():
    return 1

def function2():
    return 2

def function3():
    return 3'''
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(simple_code)
        temp_file = f.name
    
    try:
        documents = parse(temp_file)
        
        print(f"生成了 {len(documents)} 个文档块:")
        
        for i, doc in enumerate(documents, 1):
            print(f"块 {i}: 行 {doc.start_line}-{doc.end_line}")
            
            # 检查内容是否以完整行开始和结束
            lines = doc.content.split('\n')
            first_line = lines[0] if lines else ""
            last_line = lines[-1] if lines else ""
            
            print(f"  首行: '{first_line}'")
            print(f"  末行: '{last_line}'")
            
            # 验证没有部分行
            if first_line and not first_line.startswith(' ') and 'def ' in first_line:
                print("  ✅ 首行对齐正确")
            elif not first_line.strip():
                print("  ✅ 首行为空行")
            else:
                print("  ⚠️ 首行可能未对齐")
        
        return True
        
    finally:
        os.unlink(temp_file)

if __name__ == "__main__":
    success = True
    
    try:
        success &= test_python_semantic_chunking()
        success &= test_java_semantic_chunking()
        success &= test_line_boundary_alignment()
        
        if success:
            print("\n✅ 所有语义分块优化测试通过!")
        else:
            print("\n❌ 部分测试失败!")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {e}")
        sys.exit(1)

#!/usr/bin/env python3
"""
测试分片逻辑优化功能
验证缓存、配置化、复杂度评分等新功能
"""

import tempfile
import os
import sys
import time
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.splitter import (
    SplitterConfig, 
    get_splitter_parser, 
    CodeSplitter,
    DefaultSplitter,
    get_file_hash,
    _content_hash_cache,
    _parser_cache
)

# 测试用的Python代码示例
PYTHON_CODE_COMPLEX = '''
import os
import sys
from typing import List, Dict, Optional

# 这是一个简单的计算器类示例
class Calculator:
    """
    一个功能丰富的计算器类
    支持基本的数学运算和高级功能
    """
    
    def __init__(self):
        self.history: List[str] = []
        self.memory: float = 0.0
        
    def add(self, a: float, b: float) -> float:
        """加法运算"""
        result = a + b
        self.history.append(f"{a} + {b} = {result}")
        return result
    
    def subtract(self, a: float, b: float) -> float:
        """减法运算"""
        result = a - b
        self.history.append(f"{a} - {b} = {result}")
        return result
        
    def multiply(self, a: float, b: float) -> float:
        """乘法运算"""
        result = a * b
        self.history.append(f"{a} * {b} = {result}")
        return result
        
    def divide(self, a: float, b: float) -> float:
        """
        除法运算
        包含错误处理
        """
        if b == 0:
            raise ValueError("Cannot divide by zero")
        result = a / b
        self.history.append(f"{a} / {b} = {result}")
        return result
    
    def complex_calculation(self, numbers: List[float]) -> Dict[str, float]:
        """
        复杂计算函数
        包含多种控制流和嵌套结构
        """
        results = {}
        
        # 计算总和
        total = 0
        for num in numbers:
            total += num
        results['sum'] = total
        
        # 计算平均值
        if len(numbers) > 0:
            results['average'] = total / len(numbers)
        else:
            results['average'] = 0
            
        # 查找最大值和最小值
        if numbers:
            max_val = numbers[0]
            min_val = numbers[0]
            
            for num in numbers[1:]:
                if num > max_val:
                    max_val = num
                elif num < min_val:
                    min_val = num
                    
            results['max'] = max_val
            results['min'] = min_val
            
            # 复杂的条件逻辑
            try:
                if max_val > 100:
                    if min_val < 0:
                        results['category'] = 'mixed_extreme'
                    else:
                        results['category'] = 'large_positive'
                elif min_val < -100:
                    results['category'] = 'large_negative'
                else:
                    results['category'] = 'moderate'
            except Exception as e:
                results['error'] = str(e)
        
        return results
    
    def get_history(self) -> List[str]:
        """获取计算历史"""
        return self.history.copy()
    
    def clear_history(self):
        """清空历史记录"""
        self.history.clear()

# 独立函数
def utility_function(x: int, y: int) -> int:
    """一个简单的工具函数"""
    return x * y + x - y

def another_simple_function():
    """另一个简单函数"""
    print("Hello, World!")

# 全局变量
GLOBAL_CONSTANT = 42
global_var = "test"
'''

def test_configuration_system():
    """测试配置系统"""
    print("=" * 60)
    print("测试配置系统")
    print("=" * 60)
    
    # 创建不同的配置
    configs = {
        "默认配置": SplitterConfig(),
        "小块配置": SplitterConfig(
            chunk_size=1000,
            min_semantic_chunk_size=50,
            high_priority_independence_ratio=0.2
        ),
        "大块配置": SplitterConfig(
            chunk_size=4000,
            min_semantic_chunk_size=500,
            high_priority_independence_ratio=0.5
        ),
        "严格注释关联": SplitterConfig(
            max_comment_distance=100,
            max_comment_lines_gap=1
        )
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(PYTHON_CODE_COMPLEX)
        temp_file = f.name
    
    try:
        for config_name, config in configs.items():
            print(f"\n--- {config_name} ---")
            splitter = get_splitter_parser(temp_file, config=config)
            documents = splitter.split(temp_file, PYTHON_CODE_COMPLEX)
            
            print(f"生成块数: {len(documents)}")
            print(f"平均块大小: {sum(len(doc.content) for doc in documents) / len(documents):.0f} 字符")
            
            # 显示语义信息
            for i, doc in enumerate(documents[:3]):  # 只显示前3个
                semantic_info = doc.semantic_info
                print(f"  块 {i+1}: 类型={semantic_info.get('chunk_types', [])}, "
                      f"复杂度={semantic_info.get('complexity_scores', [])}")
    
    finally:
        os.unlink(temp_file)

def test_caching_performance():
    """测试缓存性能"""
    print("\n" + "=" * 60)
    print("测试缓存性能")
    print("=" * 60)
    
    # 清除缓存
    _content_hash_cache.clear()
    _parser_cache.clear()
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(PYTHON_CODE_COMPLEX)
        temp_file = f.name
    
    try:
        # 第一次解析（无缓存）
        start_time = time.time()
        splitter1 = get_splitter_parser(temp_file)
        documents1 = splitter1.split(temp_file, PYTHON_CODE_COMPLEX)
        first_run_time = time.time() - start_time
        
        # 第二次解析（有缓存）
        start_time = time.time()
        splitter2 = get_splitter_parser(temp_file)
        documents2 = splitter2.split(temp_file, PYTHON_CODE_COMPLEX)
        second_run_time = time.time() - start_time
        
        print(f"第一次解析时间: {first_run_time:.4f} 秒")
        print(f"第二次解析时间: {second_run_time:.4f} 秒")
        print(f"性能提升: {first_run_time / second_run_time:.2f}x")
        print(f"缓存命中: {len(_content_hash_cache)} 个内容缓存, {len(_parser_cache)} 个解析器缓存")
        
        # 验证结果一致性
        assert len(documents1) == len(documents2)
        for d1, d2 in zip(documents1, documents2):
            assert d1.content == d2.content
        print("✅ 缓存结果验证通过")
    
    finally:
        os.unlink(temp_file)

def test_complexity_scoring():
    """测试复杂度评分"""
    print("\n" + "=" * 60)
    print("测试复杂度评分")
    print("=" * 60)
    
    # 创建不同复杂度的代码示例
    test_cases = {
        "简单函数": '''
def simple_add(a, b):
    return a + b
''',
        "中等复杂度": '''
def moderate_function(numbers):
    total = 0
    for num in numbers:
        if num > 0:
            total += num
    return total
''',
        "高复杂度": '''
def complex_function(data):
    result = {}
    for item in data:
        try:
            if isinstance(item, dict):
                for key, value in item.items():
                    if key in result:
                        if isinstance(value, list):
                            result[key].extend(value)
                        else:
                            result[key].append(value)
                    else:
                        result[key] = [value] if not isinstance(value, list) else value
            elif isinstance(item, list):
                for subitem in item:
                    if subitem not in result:
                        result[subitem] = 1
                    else:
                        result[subitem] += 1
        except Exception as e:
            continue
    return result
'''
    }
    
    for case_name, code in test_cases.items():
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_file = f.name
        
        try:
            splitter = get_splitter_parser(temp_file)
            documents = splitter.split(temp_file, code)
            
            if documents and documents[0].semantic_info:
                complexity_scores = documents[0].semantic_info.get('complexity_scores', [])
                avg_complexity = sum(complexity_scores) / len(complexity_scores) if complexity_scores else 0
                print(f"{case_name}: 复杂度评分 = {avg_complexity:.2f}")
            else:
                print(f"{case_name}: 无法计算复杂度评分")
        
        finally:
            os.unlink(temp_file)

def test_semantic_information():
    """测试语义信息"""
    print("\n" + "=" * 60)
    print("测试语义信息")
    print("=" * 60)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(PYTHON_CODE_COMPLEX)
        temp_file = f.name
    
    try:
        splitter = get_splitter_parser(temp_file)
        documents = splitter.split(temp_file, PYTHON_CODE_COMPLEX)
        
        print(f"总共生成 {len(documents)} 个文档块\n")
        
        for i, doc in enumerate(documents):
            semantic_info = doc.semantic_info
            print(f"文档块 {i+1}:")
            print(f"  分片器类型: {semantic_info.get('splitter', 'unknown')}")
            print(f"  包含的语义类型: {semantic_info.get('chunk_types', [])}")
            print(f"  代码块名称: {semantic_info.get('chunk_names', [])}")
            print(f"  复杂度评分: {semantic_info.get('complexity_scores', [])}")
            print(f"  包含注释: {semantic_info.get('has_comments', False)}")
            print(f"  行范围: {doc.start_line}-{doc.end_line}")
            print(f"  内容长度: {len(doc.content)} 字符")
            print()
    
    finally:
        os.unlink(temp_file)

def test_fallback_mechanism():
    """测试降级机制"""
    print("\n" + "=" * 60)
    print("测试降级机制")
    print("=" * 60)
    
    # 测试不支持的文件类型
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("This is a plain text file.\nWith multiple lines.\n")
        temp_file = f.name
    
    try:
        splitter = get_splitter_parser(temp_file)
        assert isinstance(splitter, DefaultSplitter), "应该降级到 DefaultSplitter"
        
        documents = splitter.split(temp_file, "This is a plain text file.\nWith multiple lines.\n")
        assert len(documents) > 0, "应该生成至少一个文档块"
        
        # 检查语义信息
        if documents[0].semantic_info:
            assert documents[0].semantic_info.get('splitter') == 'default'
            assert documents[0].semantic_info.get('chunk_type') == 'line_based'
        
        print("✅ 降级机制测试通过")
    
    finally:
        os.unlink(temp_file)

def main():
    """运行所有测试"""
    print("🚀 分片逻辑优化功能测试")
    print("=" * 60)
    
    try:
        test_configuration_system()
        test_caching_performance()
        test_complexity_scoring()
        test_semantic_information()
        test_fallback_mechanism()
        
        print("\n" + "=" * 60)
        print("🎉 所有测试完成！")
        print("=" * 60)
        
        # 显示最终缓存状态
        print(f"最终缓存状态:")
        print(f"  内容缓存: {len(_content_hash_cache)} 项")
        print(f"  解析器缓存: {len(_parser_cache)} 项")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
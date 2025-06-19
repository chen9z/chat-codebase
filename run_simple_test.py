#!/usr/bin/env python3
"""
简单的测试脚本，验证导入路径修复是否成功
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_imports():
    """测试各种导入是否正常工作"""
    
    print("🧪 测试导入路径修复...")
    
    try:
        # 测试配置导入
        print("📦 测试配置模块导入...")
        from src.config import settings
        from src.config.parser_config import is_semantic_boundary
        print("✅ 配置模块导入成功")
        
        # 测试数据模块导入
        print("📦 测试数据模块导入...")
        from src.data.splitter import parse, Document
        print("✅ 数据模块导入成功")
        
        # 测试工具模块导入
        print("📦 测试工具模块导入...")
        from src.tools.base import BaseTool
        print("✅ 工具模块导入成功")
        
        # 测试实用工具导入
        print("📦 测试实用工具导入...")
        from src.utils.file_utils import get_file_extension
        from src.utils.logging_utils import setup_logging
        print("✅ 实用工具导入成功")
        
        # 测试一些基本功能
        print("🔧 测试基本功能...")
        
        # 测试文件扩展名检测
        ext = get_file_extension("test.java")
        assert ext == ".java", f"Expected '.java', got '{ext}'"
        print(f"   ✅ 文件扩展名检测: {ext}")
        
        # 测试语义边界检测
        is_method = is_semantic_boundary("method_declaration", "java")
        assert is_method == True, "Java method_declaration should be semantic boundary"
        print(f"   ✅ 语义边界检测: method_declaration -> {is_method}")
        
        # 测试语言映射
        java_lang = settings.ext_to_lang.get(".java")
        assert java_lang == "java", f"Expected 'java', got '{java_lang}'"
        print(f"   ✅ 语言映射: .java -> {java_lang}")
        
        print("\n🎉 所有导入测试通过！")
        print("✅ config 位置优化成功，导入路径修复完成")
        
        return True
        
    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        return False
    except AssertionError as e:
        print(f"❌ 功能测试失败: {e}")
        return False
    except Exception as e:
        print(f"❌ 未知错误: {e}")
        return False

def test_direct_execution():
    """测试直接执行模块的能力"""
    
    print("\n🧪 测试直接执行模块...")
    
    try:
        # 测试 splitter 模块的解析功能
        from src.data.splitter import parse
        
        # 创建一个简单的测试文件
        test_content = '''
public class TestClass {
    /**
     * Test method
     */
    public void testMethod() {
        System.out.println("Hello");
    }
}
'''
        
        # 创建临时文件
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.java', delete=False) as f:
            f.write(test_content)
            temp_file = f.name
        
        try:
            # 解析文件
            documents = parse(temp_file)
            print(f"   ✅ 解析成功，生成 {len(documents)} 个文档块")
            
            if documents:
                doc = documents[0]
                has_comment = "/**" in doc.content
                has_method = "testMethod" in doc.content
                print(f"   ✅ 注释和方法关联: 注释={has_comment}, 方法={has_method}")
                
                if has_comment and has_method:
                    print("   🎯 注释与方法成功关联在同一块中！")
            
        finally:
            # 清理临时文件
            os.unlink(temp_file)
        
        print("✅ 直接执行测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 直接执行测试失败: {e}")
        return False

def main():
    """主函数"""
    print("🚀 开始测试 config 位置优化结果\n")
    
    success1 = test_imports()
    success2 = test_direct_execution()
    
    print(f"\n📊 测试结果:")
    print(f"   导入测试: {'✅ 通过' if success1 else '❌ 失败'}")
    print(f"   执行测试: {'✅ 通过' if success2 else '❌ 失败'}")
    
    if success1 and success2:
        print("\n🎉 所有测试通过！config 位置优化成功！")
        print("💡 现在可以使用以下方式运行模块:")
        print("   - uv run python src/rag.py")
        print("   - uv run python run_rag.py")
        print("   - uv run python tests/unit/test_comment_association.py")
    else:
        print("\n❌ 部分测试失败，请检查配置")
        sys.exit(1)

if __name__ == "__main__":
    main()

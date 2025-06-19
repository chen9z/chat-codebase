#!/usr/bin/env python3
"""
RAG 应用运行脚本
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 现在可以正常导入
from src.rag import RAG
from src.model.llm import LLMClient

def main():
    """主函数"""
    try:
        # 检查环境变量
        if not os.getenv("OPENAI_API_BASE") or not os.getenv("OPENAI_API_KEY"):
            print("❌ 请设置 OPENAI_API_BASE 和 OPENAI_API_KEY 环境变量")
            return
        
        print("🚀 启动 RAG 应用...")
        
        # 创建 RAG 实例（使用简化的配置，避免缺少的依赖）
        llm_client = LLMClient(
            base_url=os.getenv("OPENAI_API_BASE"), 
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        app = RAG(
            llm_client=llm_client,
            model="deepseek-chat"
            # 注释掉可能缺少依赖的部分
            # embedding_model=JinaCodeEmbeddingModel(),
            # rerank_model=LocalRerankModel()
        )
        
        # 示例项目路径
        project_path = os.path.expanduser("~/workspace/spring-ai")
        if not os.path.exists(project_path):
            project_path = str(project_root)  # 使用当前项目作为示例
        
        project_name = project_path.split("/")[-1]
        
        print(f"📁 索引项目: {project_path}")
        app.index_project(project_path)
        
        print("❓ 查询示例...")
        response = app.query(project_name, "这个项目是做什么的？")
        
        print("📝 响应:")
        for chunk in response:
            print(chunk, end='', flush=True)
        print()
        
    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        print("💡 提示: 某些依赖可能未安装，请检查 pyproject.toml")
    except Exception as e:
        print(f"❌ 运行错误: {e}")

if __name__ == "__main__":
    main()

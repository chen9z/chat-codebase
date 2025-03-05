import json
import os
from typing import Dict

import dotenv
from qdrant_client import QdrantClient

from config.settings import SYSTEM_PROMPT_WITH_TOOLS
from src.data.repository import Repository
from src.model.embedding import OpenAILikeEmbeddingModel
from src.model.llm import LLMClient
from src.model.reranker import RerankAPIModel
from src.tools import (
    CodebaseSearchTool,
    GrepSearchTool,
    ListDirTool,
    ViewFileTool,
    ViewCodeItemTool,
    FindByNameTool,
    RelatedFilesTool,
)

dotenv.load_dotenv()


class Agent:
    """管理工具集合和LLM交互的类"""

    def __init__(self):
        repository = Repository(model=OpenAILikeEmbeddingModel(), vector_client=QdrantClient(path="../storage"),
                                rerank_model=RerankAPIModel())
        self.repository = repository
        # 初始化所有工具
        self.tools = {
            "codebase_search": CodebaseSearchTool(repository),
            "grep_search": GrepSearchTool(),
            "list_dir": ListDirTool(),
            "view_file": ViewFileTool(),
            "view_code_item": ViewCodeItemTool(),
            "find_by_name": FindByNameTool(),
            "related_files": RelatedFilesTool(),
        }

        # 初始化LLM客户端
        self.llm = LLMClient(base_url=os.getenv("OPENAI_API_BASE"), api_key=os.getenv("OPENAI_API_KEY"))

        # 转换工具为OpenAI格式
        self.openai_tools = self._convert_tools_to_openai_format()

    def _convert_tools_to_openai_format(self):
        """将工具转换为OpenAI tools格式"""
        openai_tools = []
        for name, tool in self.tools.items():
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": tool.description,
                    "parameters": tool.parameters
                }
            })
        return openai_tools

    def _handle_tool_calls(self, tool_calls):
        """处理工具调用返回结果"""
        results = []
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            try:
                parameters = json.loads(tool_call.function.arguments)
                result = self.execute_tool(tool_name, **parameters)
                results.append({
                    "tool_call_id": tool_call.id,
                    "output": json.dumps(result, ensure_ascii=False)
                })
            except Exception as e:
                results.append({
                    "tool_call_id": tool_call.id,
                    "output": f"工具执行错误: {str(e)}"
                })
        return results

    def get_tool_descriptions(self) -> str:
        """生成所有工具的描述信息"""
        descriptions = []
        for name, tool in self.tools.items():
            desc = f"Tool: {name}\nDescription: {tool.description}\nParameters: {json.dumps(tool.parameters, indent=2)}\n"
            descriptions.append(desc)
        return "\n".join(descriptions)

    def execute_tool(self, tool_name: str, **params) -> Dict:
        """执行指定的工具"""
        if tool_name not in self.tools:
            return {"error": f"Tool {tool_name} not found"}

        tool = self.tools[tool_name]
        try:
            result = tool.execute(**params)
            return result
        except Exception as e:
            return {"error": f"Tool execution failed: {str(e)}"}

    def query_with_tools(self, project_path: str, query: str) -> str:
        """使用工具和LLM处理查询"""
        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT_WITH_TOOLS
            },
            {
                "role": "user",
                "content": f"Project path: {project_path}\nQuery: {query}"
            }
        ]

        while True:
            response = self.llm.get_response_with_tools(
                model="gpt-4o-2024-11-20",
                messages=messages,
                tools=self.openai_tools
            )

            if not response:
                return "LLM call failed"

            print(response)
            if response["type"] == "message":
                return response["content"]

            # Handle tool calls
            tool_results = self._handle_tool_calls(response["tool_calls"])
            print(f"Tool results: {tool_results}")

            # Add tool execution results to message history
            messages.append({
                "role": "assistant",
                "tool_calls": response["tool_calls"]
            })
            messages.append({
                "role": "tool",
                "tool_call_id": tool_results[0]["tool_call_id"],
                "content": tool_results[0]["output"]
            })


def main():
    # 创建工具管理器
    agent = Agent()

    # 设置项目路径和查询
    project_path = os.path.expanduser("~/workspace/spring-ai")
    query = "spring-ai 支持哪些 LLM? 支持哪些 Embedding 模型？"

    agent.repository.index(project_path)

    # 执行查询并打印结果
    result = agent.query_with_tools(project_path, query)
    print("\nQuery result:")
    print(result)


if __name__ == "__main__":
    main()

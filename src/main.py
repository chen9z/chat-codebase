import asyncio
import json
from pathlib import Path
from typing import Dict, List
from src.tools import (
    CodebaseSearchTool,
    GrepSearchTool,
    ListDirTool,
    ViewFileTool,
    ViewCodeItemTool,
    FindByNameTool,
    RelatedFilesTool,
)
from src.model.llm import LLMClient

class ToolManager:
    """管理工具集合和LLM交互的类"""
    
    def __init__(self):
        # 初始化所有工具
        self.tools = {
            "codebase_search": CodebaseSearchTool(),
            "grep_search": GrepSearchTool(),
            "list_dir": ListDirTool(),
            "view_file": ViewFileTool(),
            "view_code_item": ViewCodeItemTool(),
            "find_by_name": FindByNameTool(),
            "related_files": RelatedFilesTool(),
        }
        
        # 初始化LLM客户端
        self.llm = LLMClient()
        
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
                    "parameters": {
                        "type": "object",
                        "properties": tool.parameters,
                        "required": list(tool.parameters.keys())
                    }
                }
            })
        return openai_tools

    async def _handle_tool_calls(self, tool_calls):
        """处理工具调用返回结果"""
        results = []
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            try:
                parameters = json.loads(tool_call.function.arguments)
                result = await self.execute_tool(tool_name, **parameters)
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

    async def execute_tool(self, tool_name: str, **params) -> Dict:
        """执行指定的工具"""
        if tool_name not in self.tools:
            return {"error": f"Tool {tool_name} not found"}
        
        tool = self.tools[tool_name]
        try:
            result = await tool.execute(**params)
            return result
        except Exception as e:
            return {"error": f"Tool execution failed: {str(e)}"}

    def query_with_tools(self, project_path: str, query: str) -> str:
        """使用工具和LLM处理查询"""
        messages = [
            {
                "role": "system",
                "content": "You are an AI assistant that helps users understand codebases. Please use the provided tools to answer user questions."
            },
            {
                "role": "user",
                "content": f"Project path: {project_path}\nQuery: {query}"
            }
        ]

        while True:
            response = self.llm.get_response_with_tools(
                messages=messages,
                tools=self.openai_tools
            )
            
            if not response:
                return "LLM call failed"
                
            if response["type"] == "message":
                return response["content"]
                
            # Handle tool calls
            tool_results = asyncio.run(self._handle_tool_calls(response["tool_calls"]))
            
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

async def main():
    # 创建工具管理器
    tool_manager = ToolManager()
    
    # 设置项目路径和查询
    project_path = "/home/looper/workspace/chat-codebase"
    query = "What tools are available in this project?"
    
    # 执行查询并打印结果
    result = tool_manager.query_with_tools(project_path, query)
    print("\nQuery result:")
    print(result)

if __name__ == "__main__":
    asyncio.run(main())

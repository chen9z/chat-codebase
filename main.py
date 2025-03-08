import os

from src.agent import Agent


def main():
    # 创建工具管理器
    agent = Agent()

    # 设置项目路径和查询
    # project_path = os.path.expanduser("~/workspace/spring-ai")
    # query = "spring-ai 支持哪些 LLM? 支持哪些 Embedding 模型？给出详细的调研"

    project_path = os.path.expanduser("~/workspace/code-agent")
    query = "这个项目是做什么的？这个项目的目录有要修改的么？"
    agent.repository.index(project_path)

    # 执行查询并打印结果
    result = agent.query_with_tools(project_path, query)
    print("\nQuery result:")
    print(result)


if __name__ == "__main__":
    main()

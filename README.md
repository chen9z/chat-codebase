# Code Search Assistant

**code-agent**，是一个基于 Agent Tools 的代理系统，主要用于代码库的管理和分析

## 功能特点

- 支持多种编程语言文件的索引和搜索
- 使用向量数据库进行语义相似度搜索
- 集成大语言模型进行智能问答
- 支持文档分块和上下文管理
- 实时流式响应

### 功能描述

1. **工具集合**：
    - 项目提供了一系列工具，用于代码库的搜索、文件查看、目录列表等操作。
    - 工具包括：
        - `CodebaseSearchTool`：代码库语义搜索工具。
        - `GrepSearchTool`：基于文本模式的搜索工具。
        - `ListDirTool`：列出目录内容。
        - `ViewFileTool`：查看文件内容。
        - `ViewCodeItemTool`：查看代码项（如函数或类）。
        - `FindByNameTool`：按名称查找文件。
        - `RelatedFilesTool`：查找相关文件。

2. **与 LLM 的交互**：
    - 项目通过 `LLMClient` 与大语言模型进行交互，可能用于自然语言查询代码库或生成代码。

3. **数据存储与检索**：
    - 使用 `QdrantClient` 作为向量数据库，结合嵌入模型和重排序模型，支持高效的数据检索。

## 安装

1. 克隆项目：

```bash
git clone [repository-url]
```

2. 安装依赖：

```bash
brew install ripgrep
cd code-agent
uv pip install .
```

3. 配置环境变量：
   创建 `.env` 文件并设置必要的环境变量：

```
OPENAI_API_BASE: <your-openai-api-base>
OPENAI_API_KEY: <your-openai-api-key>
EMBEDDING_URL: <your-embedding-url>
RERANK_URL: <your-rerank-url>
```

## 使用方法

1. 运行主程序：

```bash
python main.py
```

## 许可证

MIT License

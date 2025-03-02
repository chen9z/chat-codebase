# Code Search Assistant

一个基于向量数据库的智能代码搜索和问答助手。

## 功能特点

- 支持多种编程语言文件的索引和搜索
- 使用向量数据库进行语义相似度搜索
- 集成大语言模型进行智能问答
- 支持文档分块和上下文管理
- 实时流式响应

## 项目结构

```
project_root/
├── src/                    # 源代码目录
│   ├── core/              # 核心功能
│   ├── data/              # 数据处理
│   ├── api/               # API相关
│   └── utils/             # 工具函数
├── tests/                 # 测试目录
├── config/               # 配置文件目录
├── storage/             # 数据存储目录
└── docs/                # 文档
```

## 安装

1. 克隆项目：
```bash
git clone [repository-url]
cd code-search-assistant
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

3. 配置环境变量：
创建 `.env` 文件并设置必要的环境变量：
```
API_TIMEOUT=30
```

## 使用方法

1. 运行主程序：
```bash
python src/main.py
```

2. 可用命令：
- `index`: 索引一个项目目录
- `query`: 查询已索引的项目
- `exit`: 退出程序

## 开发说明

- 使用Python 3.8+
- 遵循PEP 8编码规范
- 使用类型提示
- 保持代码模块化和可测试性

## 许可证

MIT License

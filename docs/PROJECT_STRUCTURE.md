# 项目结构说明

## 📁 推荐的项目结构

```
chat-codebase/
├── README.md                    # 项目说明
├── pyproject.toml              # 项目配置和依赖
├── uv.lock                     # 依赖锁定文件
├── .env                        # 环境变量配置
├── .gitignore                  # Git 忽略文件
│
├── src/                        # 源代码目录
│   ├── __init__.py
│   ├── main.py                 # 主入口文件
│   ├── agent.py                # 主要的 Agent 类
│   ├── rag.py                  # RAG 相关功能
│   │
│   ├── data/                   # 数据处理模块
│   │   ├── __init__.py
│   │   ├── repository.py       # 数据仓库
│   │   └── splitter.py         # 代码分割器（含注释关联）
│   │
│   ├── model/                  # 模型相关
│   │   ├── __init__.py
│   │   ├── embedding.py        # 嵌入模型
│   │   ├── llm.py             # 大语言模型
│   │   └── reranker.py        # 重排序模型
│   │
│   ├── tools/                  # 工具集合
│   │   ├── __init__.py
│   │   ├── base.py            # 基础工具类
│   │   ├── codebase_search.py # 代码库搜索
│   │   ├── grep_search.py     # 文本搜索
│   │   ├── view_code_item.py  # 代码项查看
│   │   └── ...                # 其他工具
│   │
│   ├── utils/                  # 工具函数
│   │   ├── __init__.py
│   │   ├── file_utils.py      # 文件操作工具
│   │   ├── text_utils.py      # 文本处理工具
│   │   └── logging_utils.py   # 日志工具
│   │
│   └── config/                 # 配置模块
│       ├── __init__.py
│       ├── settings.py        # 基础设置
│       └── parser_config.py   # 解析器配置
│
├── tests/                      # 测试目录
│   ├── __init__.py
│   ├── conftest.py            # pytest 配置
│   │
│   ├── unit/                  # 单元测试
│   │   ├── __init__.py
│   │   ├── test_splitter.py   # 分割器测试
│   │   ├── test_comment_association.py  # 注释关联测试
│   │   └── test_tools.py      # 工具测试
│   │
│   ├── integration/           # 集成测试
│   │   ├── __init__.py
│   │   ├── test_end_to_end.py # 端到端测试
│   │   └── test_agent.py      # Agent 集成测试
│   │
│   └── fixtures/              # 测试数据
│       ├── sample_code/       # 示例代码文件
│       └── test_data.json     # 测试数据
│
├── docs/                       # 文档目录
│   ├── PROJECT_STRUCTURE.md   # 项目结构说明
│   ├── API.md                 # API 文档
│   ├── DEVELOPMENT.md         # 开发指南
│   └── DEPLOYMENT.md          # 部署指南
│
└── scripts/                   # 脚本目录
    ├── setup.sh              # 环境设置脚本
    ├── run_tests.sh          # 测试运行脚本
    └── build.sh              # 构建脚本
```

## 🔧 结构优化要点

### 1. **清晰的模块分离**
- `src/` - 所有源代码（包括配置）
- `tests/` - 测试代码
- `docs/` - 文档
- `scripts/` - 脚本

### 2. **完整的包结构**
- 每个目录都有 `__init__.py`
- 支持相对导入和绝对导入

### 3. **测试组织**
- `unit/` - 单元测试
- `integration/` - 集成测试
- `fixtures/` - 测试数据

### 4. **配置管理**
- 环境变量在 `.env`
- 代码配置在 `src/config/`
- 项目配置在 `pyproject.toml`

## 🚀 使用方式

### 运行测试
```bash
# 运行所有测试
uv run pytest

# 运行单元测试
uv run pytest tests/unit/

# 运行集成测试
uv run pytest tests/integration/

# 运行特定测试
uv run pytest tests/unit/test_comment_association.py
```

### 导入模块
```python
# 从项目根目录导入
from src.data.splitter import parse
from src.tools.codebase_search import CodebaseSearchTool
from src.config.settings import ext_to_lang
```

## 📝 迁移步骤

1. 移动测试文件到 `tests/` 目录
2. 创建缺少的 `__init__.py` 文件
3. 修复导入路径
4. 更新 `pyproject.toml` 中的测试路径
5. 清理根目录的临时文件

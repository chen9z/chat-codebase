# 测试目录结构

本目录包含项目的所有测试文件，按照功能和类型进行组织。

## 📁 目录结构

```
tests/
├── unit/                           # 单元测试
│   ├── data/                       # 数据处理相关测试
│   │   ├── splitters/              # 语义分块器测试
│   │   │   ├── test_enhanced_semantic_splitter.py
│   │   │   ├── test_priority_merging.py
│   │   │   ├── test_continuous_chunking.py
│   │   │   └── test_truly_continuous.py
│   │   └── test_comment_association.py
│   └── config/                     # 配置相关测试
├── integration/                    # 集成测试
│   └── semantic_chunking/          # 语义分块集成测试
│       └── test_rag_optimization.py
├── demos/                          # 演示和示例
│   └── demo_semantic_chunking.py
└── debug/                          # 调试和诊断工具
    └── debug_line_numbers.py
```

## 🧪 测试类型说明

### 单元测试 (unit/)
测试单个模块或函数的功能，不依赖外部系统。

- **data/splitters/**: 语义分块器的核心功能测试
  - `test_enhanced_semantic_splitter.py`: 增强语义分块器测试
  - `test_priority_merging.py`: 优先级合并逻辑测试
  - `test_continuous_chunking.py`: 连续分块测试
  - `test_truly_continuous.py`: 真正连续分块测试

### 集成测试 (integration/)
测试多个模块协同工作的功能。

- **semantic_chunking/**: 语义分块的端到端测试
  - `test_rag_optimization.py`: RAG优化的集成测试

### 演示 (demos/)
展示功能的示例代码，可以作为使用指南。

- `demo_semantic_chunking.py`: 语义分块功能演示

### 调试工具 (debug/)
用于问题诊断和调试的工具。

- `debug_line_numbers.py`: 行号计算调试工具

## 🚀 运行测试

### 运行所有测试
```bash
./scripts/run_tests.sh
```

### 运行特定类型的测试
```bash
# 只运行单元测试
uv run python -m pytest tests/unit/ -v

# 只运行集成测试
uv run python -m pytest tests/integration/ -v

# 只运行语义分块测试
uv run python -m pytest tests/unit/data/splitters/ -v
```

### 运行演示
```bash
# 运行语义分块演示
uv run python tests/demos/demo_semantic_chunking.py

# 运行测试时包含演示
./scripts/run_tests.sh --with-demos
```

### 运行调试工具
```bash
# 调试行号计算
uv run python tests/debug/debug_line_numbers.py
```

## 📊 测试覆盖的功能

### 语义分块核心功能
- ✅ Tree-sitter解析器集成
- ✅ 语义节点识别和分类
- ✅ 注释与代码关联
- ✅ 优先级引导的合并策略
- ✅ 连续分块算法
- ✅ 行边界对齐

### RAG优化功能
- ✅ 元数据提取
- ✅ 检索优化
- ✅ 上下文生成
- ✅ 关系建立

### 多语言支持
- ✅ Python语义分块
- ✅ Java语义分块
- ✅ JavaScript/TypeScript支持
- ✅ 其他语言的基础支持

## 🔧 添加新测试

### 单元测试
在 `tests/unit/` 下创建对应模块的测试文件：
```python
#!/usr/bin/env python3
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Your test code here
```

### 集成测试
在 `tests/integration/` 下创建功能相关的测试：
```python
# 测试多个模块的协同工作
```

### 演示代码
在 `tests/demos/` 下创建功能演示：
```python
# 展示如何使用某个功能
```

## 📝 测试命名规范

- 测试文件：`test_*.py`
- 演示文件：`demo_*.py`
- 调试工具：`debug_*.py`
- 测试函数：`test_*()` 或 `def test_*():`
- 演示函数：`demo_*()` 或 `main()`

## 🎯 测试质量标准

- ✅ 每个测试应该独立运行
- ✅ 测试应该有清晰的断言
- ✅ 测试应该包含正面和负面用例
- ✅ 测试应该有适当的文档说明
- ✅ 集成测试应该测试真实场景

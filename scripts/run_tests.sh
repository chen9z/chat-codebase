#!/bin/bash

# 测试运行脚本

set -e

echo "🧪 运行项目测试..."

# 确保在项目根目录
cd "$(dirname "$0")/.."

echo "📍 当前目录: $(pwd)"

# 运行单元测试
echo "🔬 运行单元测试..."
uv run python -m pytest tests/unit/ -v

# 运行集成测试
echo "🔗 运行集成测试..."
uv run python -m pytest tests/integration/ -v

# 运行语义分块专项测试
echo "📊 运行语义分块测试..."
uv run python -m pytest tests/unit/data/splitters/ -v

# 运行演示测试（可选）
if [ "$1" = "--with-demos" ]; then
    echo "🎭 运行演示测试..."
    uv run python tests/demos/demo_semantic_chunking.py
fi

echo "✅ 所有测试完成！"

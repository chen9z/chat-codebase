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

echo "✅ 所有测试完成！"

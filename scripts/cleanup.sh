#!/bin/bash

# 项目清理脚本

set -e

echo "🧹 清理项目文件..."

# 确保在项目根目录
cd "$(dirname "$0")/.."

echo "📍 当前目录: $(pwd)"

# 清理 Python 缓存文件
echo "🗑️  清理 Python 缓存文件..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true
find . -type f -name "*.pyo" -delete 2>/dev/null || true

# 清理测试缓存
echo "🗑️  清理测试缓存..."
rm -rf .pytest_cache 2>/dev/null || true
rm -rf .coverage 2>/dev/null || true
rm -rf htmlcov 2>/dev/null || true

# 清理构建文件
echo "🗑️  清理构建文件..."
rm -rf build 2>/dev/null || true
rm -rf dist 2>/dev/null || true
rm -rf *.egg-info 2>/dev/null || true

# 清理临时文件
echo "🗑️  清理临时文件..."
find . -type f -name "*.tmp" -delete 2>/dev/null || true
find . -type f -name ".DS_Store" -delete 2>/dev/null || true

# 移除根目录的测试文件（如果存在）
echo "🗑️  移除根目录的旧测试文件..."
rm -f test_comment_association.py 2>/dev/null || true
rm -f detailed_test.py 2>/dev/null || true
rm -f simple_test.py 2>/dev/null || true

echo "✅ 清理完成！"

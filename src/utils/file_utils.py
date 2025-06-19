"""
文件操作工具函数
"""

import os
import tempfile
from pathlib import Path
from typing import List, Optional, Union


def get_file_extension(file_path: Union[str, Path]) -> str:
    """获取文件扩展名"""
    return Path(file_path).suffix.lower()


def is_code_file(file_path: Union[str, Path]) -> bool:
    """判断是否为代码文件"""
    code_extensions = {
        '.py', '.java', '.js', '.ts', '.jsx', '.tsx',
        '.go', '.rs', '.cpp', '.c', '.h', '.hpp',
        '.cs', '.php', '.rb', '.swift', '.kt', '.scala'
    }
    return get_file_extension(file_path) in code_extensions


def read_file_safe(file_path: Union[str, Path], encoding: str = 'utf-8') -> Optional[str]:
    """安全读取文件内容"""
    try:
        with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
            return f.read()
    except Exception:
        return None


def write_file_safe(file_path: Union[str, Path], content: str, encoding: str = 'utf-8') -> bool:
    """安全写入文件内容"""
    try:
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w', encoding=encoding) as f:
            f.write(content)
        return True
    except Exception:
        return False


def find_files_by_extension(directory: Union[str, Path], extensions: List[str]) -> List[Path]:
    """根据扩展名查找文件"""
    directory = Path(directory)
    files = []
    
    for ext in extensions:
        if not ext.startswith('.'):
            ext = '.' + ext
        files.extend(directory.rglob(f'*{ext}'))
    
    return files


def create_temp_file(content: str, suffix: str = '.tmp') -> str:
    """创建临时文件"""
    with tempfile.NamedTemporaryFile(mode='w', suffix=suffix, delete=False) as f:
        f.write(content)
        return f.name


def cleanup_temp_file(file_path: str) -> bool:
    """清理临时文件"""
    try:
        os.unlink(file_path)
        return True
    except Exception:
        return False


def get_project_root() -> Path:
    """获取项目根目录"""
    current = Path(__file__).parent
    while current.parent != current:
        if (current / 'pyproject.toml').exists():
            return current
        current = current.parent
    return Path.cwd()


def ensure_directory(directory: Union[str, Path]) -> bool:
    """确保目录存在"""
    try:
        Path(directory).mkdir(parents=True, exist_ok=True)
        return True
    except Exception:
        return False

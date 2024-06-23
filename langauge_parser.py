import os

def is_support_file(file_path: str) -> bool:
    return os.path.splitext(file_path)[1] in [".java", ".xml", "yml", ".yaml", ".md"]

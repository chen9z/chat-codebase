from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
STORAGE_DIR = BASE_DIR / "storage"

# Model configurations
EMBEDDING_MODEL = {
    "local": {
        "model_name": "jinaai/jina-reranker-v2-base-multilingual",
        "model_args": {"torch_dtype": "auto"},
    }
}

RERANK_MODEL = {
    "local": {
        "model_name": "jinaai/jina-reranker-v2-base-multilingual",
        "model_args": {"torch_dtype": "auto"},
    }
}

LLM_MODEL = {
    "default": "qwen2.5-coder:14b-instruct-q6_K",
}

# Vector DB configurations
VECTOR_DB = {
    "path": str(STORAGE_DIR),
}

# API configurations
API_SETTINGS = {
    "timeout": int(os.getenv("API_TIMEOUT", 30)),
}

# System prompts
SYSTEM_PROMPT = """
You are a large language AI assistant. You are given a user question, and please write clean, concise and accurate answer to the question.
Your answer must be correct, accurate and written by an expert using an unbiased and professional tone. Please limit to 1024 tokens.
Do not give any information that is not related to the question, and do not repeat.
your answer must be written with **Chinese**
"""

USER_PROMPT_TEMPLATE = """
Here are the set of contexts:

{context}

Remember, don't blindly repeat the contexts verbatim.
###
And here is the user question:
{query}

###
Answer:
"""

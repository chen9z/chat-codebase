import os
import sys
from pathlib import Path
from typing import Generator, Optional, List

# Add project root to Python path when running as script
if __name__ == '__main__':
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

from qdrant_client import QdrantClient

from src.config import settings
from src.model.embedding import JinaCodeEmbeddingModel, OpenAILikeEmbeddingModel
from src.model.llm import LLMClient
from src.model.reranker import RerankModel, LocalRerankModel, RerankAPIModel
from src.data.repository import Repository
from src.data.splitter import Document


class RAG:
    """Main application class for code search and QA."""

    def __init__(
            self,
            llm_client: Optional[LLMClient] = None,
            model: str = "deepseek-chat",
            embedding_model=None,
            rerank_model: Optional[RerankModel] = None,
            vector_store_path: str = "./storage"
    ):
        """Initialize the application with its components.
        
        Args:
            embedding_model: Optional custom embedding model
            rerank_model: Optional custom rerank model
            vector_store_path: Path to vector store
        """
        self.llm_client = llm_client or LLMClient()
        self.model = model
        self.embedding_model = embedding_model or JinaCodeEmbeddingModel()
        self.rerank_model = rerank_model or LocalRerankModel()
        self.vector_client = QdrantClient(path=vector_store_path)
        self.repository = Repository(
            self.embedding_model,
            self.vector_client,
            rerank_model=rerank_model or LocalRerankModel()
        )

    def index_project(self, project_path: str) -> None:
        """Index a project directory.
        
        Args:
            project_path: Path to the project directory
        """
        project_path = Path(project_path).expanduser()
        self.repository.index(str(project_path))
        print(f"Successfully indexed project: {project_path.name}")

    def format_context(self, documents: List[Document]) -> str:
        if not documents:
            return ""

        format_context = ""
        for doc in documents:
            format_context += f"file:///{doc.path} \n" + doc.content
        return format_context

    def query(self, project_name: str, query: str) -> Generator:
        """Query the project and get responses.
        
        Args:
            project_name: Name of the project to search in
            query: User's question
            
        Returns:
            Generator yielding response chunks
        """
        # Search for relevant documents
        results = self.repository.search(project_name, query)

        # Format context from search results
        context = self.format_context(results)
        print("Context:", context)
        messages = [
            {
                "role": "system",
                "content": settings.SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": settings.USER_PROMPT_TEMPLATE.format(context=context, query=query)
            }
        ]

        print("Messages:", messages)

        # Get streaming response
        return self.llm_client.get_response(model=self.model, messages=messages, stream=True)


if __name__ == '__main__':
    try:
        print("🚀 启动 RAG 应用...")

        # 检查环境变量
        if not os.getenv("OPENAI_API_BASE") or not os.getenv("OPENAI_API_KEY"):
            print("❌ 请设置 OPENAI_API_BASE 和 OPENAI_API_KEY 环境变量")
            exit(1)

        # 创建 RAG 实例
        app = RAG(
            llm_client=LLMClient(
                base_url=os.getenv("OPENAI_API_BASE"),
                api_key=os.getenv("OPENAI_API_KEY")
            ),
            model="deepseek-chat",
            embedding_model=JinaCodeEmbeddingModel(),
            rerank_model=LocalRerankModel()
        )

        project_path = os.path.expanduser("~/workspace/spring-ai")
        if not os.path.exists(project_path):
            # 如果示例路径不存在，使用当前项目
            project_path = str(Path(__file__).parent.parent)

        project_name = project_path.split("/")[-1]

        print(f"📁 索引项目: {project_path}")
        app.index_project(project_path)

        print("❓ 查询: spring ai 是什么？")
        response = app.query(project_name, "spring ai 是什么？")

        print("📝 响应:")
        for chunk in response:
            print(chunk, end='', flush=True)
        print()

    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        print("💡 提示: 请使用 'uv run python src/rag.py' 或安装缺少的依赖")
    except Exception as e:
        print(f"❌ 运行错误: {e}")

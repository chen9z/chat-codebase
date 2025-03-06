import os
from pathlib import Path
from typing import Generator, Optional, List

from qdrant_client import QdrantClient

import config.settings
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
                "content": config.settings.SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": config.settings.USER_PROMPT_TEMPLATE.format(context=context, query=query)
            }
        ]

        print("Messages:", messages)

        # Get streaming response
        return self.llm_client.get_response(model=self.model, messages=messages, stream=True)


if __name__ == '__main__':
    app = RAG(llm_client=LLMClient(base_url=os.getenv("OPENAI_API_BASE"), api_key=os.getenv("OPENAI_API_KEY")),
              model="deepseek-chat",
              embedding_model=OpenAILikeEmbeddingModel(),
              rerank_model=RerankAPIModel())

    project_path = os.path.expanduser("~/workspace/spring-ai")
    project_name = project_path.split("/")[-1]
    app.index_project(project_path)
    response = app.query(project_name, "spring ai 是什么？")
    print("Response:", response)

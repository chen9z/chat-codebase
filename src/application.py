from pathlib import Path
from typing import Generator, Optional, List

from qdrant_client import QdrantClient

import config.settings
from src.model.embedding import JinaCodeEmbeddingModel
from src.model.llm import LLMClient
from src.model.reranker import RerankModel, LocalRerankModel
from src.data.repository import Repository
from src.data.splitter import Document


class Application:
    """Main application class for code search and QA."""

    def __init__(
            self,
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
        self.embedding_model = embedding_model or JinaCodeEmbeddingModel()
        self.rerank_model = rerank_model or LocalRerankModel()
        self.vector_client = QdrantClient(path=vector_store_path)
        self.repository = Repository(
            self.embedding_model,
            self.vector_client,
            rerank_model=rerank_model or LocalRerankModel()
        )
        self.llm_client = LLMClient()

    def index_project(self, project_path: str) -> None:
        """Index a project directory.
        
        Args:
            project_path: Path to the project directory
        """
        project_path = Path(project_path).expanduser()
        self.repository.index(str(project_path))
        print(f"Successfully indexed project: {project_path.name}")

    def format_context(self, documents: List[Document]) -> str:
        """Format search results into context for LLM input.
        
        This method formats the search results in a way that's optimal for LLM understanding,
        including source information, relevance scores, and clear document separation.
        
        Args:
            documents: List of Document objects from search results
            
        Returns:
            Formatted context string ready for LLM consumption
        """
        if not documents:
            return ""

        context_parts = []
        for doc in documents:
            # Format each document with its metadata and content
            context_parts.append(
                f"Source: {doc.path}\n"
                f"Relevance Score: {doc.score:.2f}\n"
                f"Content:\n{doc.content}\n"
            )

        # Join all parts with clear separation
        return "\n---\n".join(context_parts)

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
                "content": config.settings.USER_PROMPT_TEMPLATE.format(context, query)
            }
        ]

        print("Messages:", messages)

        # Get streaming response
        return self.llm_client.get_response(messages)

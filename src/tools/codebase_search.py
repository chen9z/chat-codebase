import os
from typing import Any, Dict, Optional

from qdrant_client import QdrantClient

from src.data.repository import Repository
from src.model.embedding import OpenAILikeEmbeddingModel
from src.model.reranker import RerankAPIModel
from src.tools.base import BaseTool


class CodebaseSearchTool(BaseTool):
    """Tool for performing semantic search over codebase using Repository."""

    @property
    def name(self) -> str:
        return "codebase_search"

    def __init__(
            self,
            repository: Optional[Repository] = None,
    ):
        self.repository = Repository(
            model=repository.model,
            vector_client=repository.vector_client,
            rerank_model=repository.rerank_model,
        )

    @property
    def description(self) -> str:
        return """Find snippets of code from the codebase most relevant to the search query.
        This performs best when the search query is more precise and relating to the function
        or purpose of code."""

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "project_name": {
                    "type": "string",
                    "description": "Name of the project to search in"
                },
                "query": {
                    "type": "string",
                    "description": "Search query"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results to return",
                    "default": 5
                }
            },
            "required": ["project_name", "query"]
        }

    def execute(
            self,
            project_name: str,
            query: str,
            limit: int = 5
    ) -> Dict[str, Any]:
        """Execute semantic search over codebase using Repository.
        
        Args:
            project_name: Name of the project to search in
            query: The search query
            limit: Maximum number of results to return
            
        Returns:
            Dict containing search results with relevant code snippets
        """
        try:
            # 使用 Repository 执行搜索
            results = self.repository.search(
                project_name=project_name,
                query=query,
                limit=limit
            )

            # 格式化搜索结果
            matches = []
            for doc in results:
                matches.append({
                    "file": doc.path,
                    "chunk_id": doc.chunk_id,
                    "content": doc.content,
                    "score": doc.score,
                    "start_line": doc.start_line,
                    "end_line": doc.end_line
                })

            return {
                "matches": matches,
                "total_results": len(matches)
            }

        except Exception as e:
            print(e)
            return {
                "error": f"Search failed: {str(e)}",
                "matches": []
            }

    def index_project(self, project_dir: str) -> None:
        """Index a project directory for searching.
        
        Args:
            project_dir: Path to the project directory to index
        """
        try:
            self.repository.index(project_dir)
        except Exception as e:
            raise Exception(f"Failed to index project: {str(e)}")


async def main():
    project_dir = os.path.expanduser("~/workspace/spring-ai")
    qdrant_client = QdrantClient(path="./storage")
    tool = CodebaseSearchTool(embedding_model=OpenAILikeEmbeddingModel(), rerank_model=RerankAPIModel(),
                              vector_client=qdrant_client)
    tool.index_project(project_dir)
    result = tool.execute(project_dir.split("/")[-1], 'spring ai 是什么？', 5)
    print(result)


if __name__ == '__main__':
    main()

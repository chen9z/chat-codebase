import os.path
import uuid
from typing import List

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance

from src.data import splitter
from src.data.splitter import Document
from src.model.embedding import EmbeddingModel
from src.model.reranker import RerankModel, LocalRerankModel

IGNORED_DIRS = {".git", ".idea", ".vscode", "__pycache__", "venv", "node_modules", ".venv", ".pytest_cache"}
SUPPORTED_EXTENSIONS = {
    ".java", ".xml", ".yml", ".yaml", ".properties", ".sql", ".md",
    ".js", ".ts", ".css", ".html", ".vue", ".py", ".go"
}


class Repository:
    def __init__(
            self,
            model: EmbeddingModel,
            vector_client: QdrantClient,
            rerank_model: RerankModel,
    ):
        self.model = model
        self.vector_client = vector_client
        self.embedding_dimension = self.model.get_embedding_dimension()
        self.rerank_model = rerank_model or LocalRerankModel()

    def index(self, project_dir: str) -> None:
        # get the absolute path
        project_dir = os.path.expanduser(project_dir)
        if not os.path.exists(project_dir):
            raise FileNotFoundError(f"project_dir not found at {project_dir}")
        if not os.path.isdir(project_dir):
            raise FileNotFoundError(f"project_dir is not a directory at {project_dir}")

        project_name = os.path.basename(project_dir)

        if self.vector_client.collection_exists(collection_name=project_name):
            print(f"Project {project_name} already exists")
            return

        self.vector_client.create_collection(
            collection_name=project_name,
            vectors_config=VectorParams(size=self.embedding_dimension, distance=Distance.COSINE),
        )

        points = []
        file_paths = traverse_files(project_dir)
        for file_path in file_paths:
            documents = splitter.parse(file_path)
            for document in documents:
                embeddings = self.model.get_embedding(document.content)
                point = PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embeddings,
                    payload=document.__dict__
                )
                points.append(point)

        if points:
            operation = self.vector_client.upsert(
                collection_name=project_name,
                points=points,
                wait=True
            )
            if operation.status == "completed":
                print("Data inserted successfully")
            else:
                print("Data insertion failed")

    def search(self, project_name: str, query: str, limit: int = 10) -> List[Document]:
        # First stage: Vector similarity search
        q_embeddings = self.model.get_embedding(query)
        search_results = self.vector_client.search(
            os.path.basename(project_name),  # 模型调用时，会传入绝对路径，取目录名
            q_embeddings,
            limit=limit * 2
        )

        if not search_results:
            return []

        documents = [Document(**result.payload) for result in search_results]
        # Second stage: Rerank
        reranked_docs = self.rerank_model.rerank(query, documents, limit=limit)
        return reranked_docs


def traverse_files(dir_path: str):
    """
    Traverses all files in the given directory, skipping ignored directories.
    Returns only files with supported extensions.
    """
    for root, dirs, files in os.walk(dir_path, topdown=True):
        # Modify dirs in-place to skip ignored directories efficiently
        dirs[:] = [d for d in dirs if d not in IGNORED_DIRS]

        for file in files:
            file_path = os.path.join(root, file)
            if os.path.splitext(file_path)[1] in SUPPORTED_EXTENSIONS:
                yield file_path


def is_support_file(file_path: str) -> bool:
    """
    Checks if a file has a supported extension.
    Note: Directory filtering is now handled in traverse_files.
    """
    return os.path.splitext(file_path)[1] in SUPPORTED_EXTENSIONS


if __name__ == '__main__':
    files = traverse_files(os.path.expanduser("~/workspace/code-agent"))
    for file in files:
        print(file)

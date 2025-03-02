import os.path
from typing import Optional, List

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance

from src.data import splitter
from src.model.embedding import EmbeddingModel
from src.model.reranker import RerankModel, LocalRerankModel
from src.data.splitter import is_support_file, Document


class Repository:
    def __init__(
            self,
            model: EmbeddingModel,
            vector_client: QdrantClient,
            rerank_model: Optional[RerankModel] = None,
            persist_dir: str = "./storage"
    ):
        self.persist_dir = persist_dir
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
            if is_support_file(file_path):
                documents = splitter.parse(file_path)
                for document in documents:
                    embeddings = self.model.get_embedding(document.content)
                    point = PointStruct(
                        id=document.chunk_id,
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
            project_name,
            q_embeddings,
            limit=min(100, limit * 4)  # Fetch more candidates for reranking
        )

        if not search_results:
            return []

        documents = [Document(**result.payload) for result in search_results]
        reranked_docs = self.rerank_model.rerank(query, documents, limit=limit)

        return reranked_docs


def traverse_files(dir_path: str):
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            yield os.path.join(root, file)

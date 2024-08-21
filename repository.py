import os.path

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance

import splitter
from embedding_model import EmbeddingModel
from embedding_model import OpenAILikeEmbeddingModel
from splitter import is_support_file, Document


class Index:
    def __init__(self, model: EmbeddingModel, vector_client: QdrantClient, persist_dir="./storage"):
        self.persist_dir = persist_dir
        self.model = model
        self.vector_client = vector_client
        self.embedding_dimension = self.model.get_embedding_dimension()

    def encode(self, project_dir: str):
        # get the absolute path
        project_dir = os.path.expanduser(project_dir)
        if not os.path.exists(project_dir):
            raise FileNotFoundError(f"project_dir not found at {project_dir}")
        if not os.path.isdir(project_dir):
            raise FileNotFoundError(f"project_dir is not a directory at {project_dir}")

        project_name = project_dir.split("/")[-1]

        if self.vector_client.collection_exists(collection_name=project_name):
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
                    # print(f"=====encode file {document.path}")
                    embeddings = self.model.get_embedding(document.content)
                    point = PointStruct(id=document.chunk_id, vector=embeddings,
                                        payload=document.__dict__)
                    points.append(point)
        operation = self.vector_client.upsert(collection_name=project_name, points=points, wait=True)
        if operation.status == "completed":
            print("Data inserted successfully")
        else:
            print("Data inserted Failed")

    def query_documents(self, project_dir, query: str, limit=10) -> list[Document]:
        q_embeddings = self.model.get_embedding(query)
        query_result = self.vector_client.search(project_dir, q_embeddings, limit=limit)
        documents = []
        for point in query_result:
            doc = Document(**point.payload)
            doc.score = point.score
            documents.append(doc)
        return documents


def traverse_files(dir_path: str):
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            yield os.path.join(root, file)


def get_index() -> Index:
    model = OpenAILikeEmbeddingModel()
    vector_client = QdrantClient(path="./storage")
    index = Index(model, vector_client)
    return index

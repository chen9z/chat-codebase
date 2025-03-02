from abc import ABC, abstractmethod
from typing import List

from openai import OpenAI
from sentence_transformers import SentenceTransformer


class EmbeddingModel(ABC):
    """Abstract base class for embedding models."""

    @abstractmethod
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for a text."""
        pass

    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embedding vector."""
        pass


class JinaCodeEmbeddingModel(EmbeddingModel):
    """Local embedding model implementation using sentence-transformers."""

    def __init__(self):
        self.model = SentenceTransformer(model_name_or_path="jinaai/jina-embeddings-v2-base-code",
                                         trust_remote_code=True)

    def get_embedding(self, text: str) -> List[float]:
        return self.model.encode(text, show_progress_bar=True).tolist()

    def get_embedding_dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()


class OpenAILikeEmbeddingModel(EmbeddingModel):

    def __init__(self):
        super().__init__()
        self.client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="host")
        self.embedding_dimension = 768

    def get_embedding_dimension(self):
        return self.embedding_dimension

    def get_embedding(self, text: str) -> List[float]:
        response = self.client.embeddings.create(model="", input=[text])
        return response.data[-1].embedding

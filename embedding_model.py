from typing import List

from openai import OpenAI
from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    def __init__(self):
        super().__init__()

    def get_embedding_dimension(self):
        pass

    def get_embedding(self, text: str) -> List[float]:
        pass


class LocalEmbeddingModel(EmbeddingModel):
    def __init__(self):
        super().__init__()
        self.model = SentenceTransformer(model_name_or_path="jinaai/jina-embeddings-v2-base-code",
                                         trust_remote_code=True)

    def get_embedding_dimension(self):
        return self.model.get_sentence_embedding_dimension()

    def get_embedding(self, text: str) -> List[float]:
        return self.model.encode(text, show_progress_bar=True).tolist()


class OpenAILikeEmbeddingModel(EmbeddingModel):

    def __init__(self):
        super().__init__()
        self.client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="host")
        self.embedding_dimension = 768

    def get_embedding_dimension(self):
        return self.embedding_dimension

    def get_embedding(self, text: str) -> List[float]:
        response = self.client.embeddings.create(model="", input=[text])
        return response.data[0].embedding


if __name__ == '__main__':
    open_like_model = OpenAILikeEmbeddingModel()
    print(open_like_model.get_embedding_dimension())
    print(open_like_model.get_embedding("hello world"))

    local_model = LocalEmbeddingModel()
    print(local_model.get_embedding_dimension())
    print(local_model.get_embedding("hello world"))

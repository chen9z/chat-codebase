from abc import ABC, abstractmethod
from typing import List

import requests
from transformers import AutoModelForSequenceClassification

from src.data.splitter import Document


class RerankModel(ABC):
    """Abstract base class for rerank models."""

    @abstractmethod
    def rerank(self, query: str, documents: List[Document], limit: int = 10) -> List[Document]:
        """Rerank documents based on their relevance to the query."""
        pass


class RerankAPIModel(RerankModel):
    """Jina AI rerank model implementation."""

    def __init__(self):
        pass

    def rerank(self, query: str, documents: List[Document], limit: int = 10) -> List[Document]:
        headers = {"Content-Type": "application/json"}
        try:
            response = requests.post("http://10.245.29.174:8080/v1/rerank",
                                     json={
                                         "query": query,
                                         "documents": [doc.content for doc in documents],
                                         "top_n": limit
                                     },
                                     headers=headers)
            response.raise_for_status()
            rankings = response.json()
            reranked_docs = []
            for rank in rankings:
                doc = documents[rank["index"]]
                doc.score = rank["score"]
                reranked_docs.append(doc)

            return reranked_docs
        except Exception as e:
            print(e)
            raise e


class LocalRerankModel(RerankModel):
    """Local rerank model implementation using sentence-transformers."""

    def __init__(self):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            'jinaai/jina-reranker-v2-base-multilingual',
            torch_dtype="auto",
            trust_remote_code=True,
        )

    def rerank(self, query: str, documents: List[Document], limit: int = 10) -> List[Document]:
        sentence_pairs = [[query, doc] for doc in documents]
        reranked_docs = self.model.rerank(sentence_pairs, max_length=8096)
        return reranked_docs


if __name__ == '__main__':
    query = "如何在Python中创建一个列表"
    documents = [
        Document(content="创建一个列表的Python代码示例"),
    ]

    reranker = RerankAPIModel()
    reranked_docs = reranker.rerank(query, documents, 10)
    for doc in reranked_docs:
        print(doc)

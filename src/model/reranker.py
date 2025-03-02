from abc import ABC, abstractmethod
from typing import List, Dict

import requests
from transformers import AutoModelForSequenceClassification

from config.settings import RERANK_MODEL


class RerankModel(ABC):
    """Abstract base class for rerank models."""

    @abstractmethod
    def rerank(self, query: str, documents: List[Dict], limit: int = 10) -> List[Dict]:
        """Rerank documents based on their relevance to the query."""
        pass


class RerankAPIModel(RerankModel):
    """Jina AI rerank model implementation."""

    def __init__(self):
        config = RERANK_MODEL["jina"]
        self.api_key = config["api_key"]
        self.api_url = config["api_url"]

    def rerank(self, query: str, documents: List[Dict], limit: int = 10) -> List[Dict]:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        data = {
            "query": query,
            "documents": [doc["content"] for doc in documents],
            "limit": limit
        }

        response = requests.post(self.api_url, headers=headers, json=data)
        response.raise_for_status()

        results = response.json()["results"]
        reranked_docs = []
        for result in results:
            doc_idx = result["index"]
            reranked_docs.append({
                **documents[doc_idx],
                "score": result["score"]
            })
        return reranked_docs


class LocalRerankModel(RerankModel):
    """Local rerank model implementation using sentence-transformers."""

    def __init__(self):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            'jinaai/jina-reranker-v2-base-multilingual',
            torch_dtype="auto",
            trust_remote_code=True,
        )

    def rerank(self, query: str, documents: List[dict], limit: int = 10) -> List[Dict]:
        sentence_pairs = [[query, doc] for doc in documents]
        reranked_docs = self.model.rerank(sentence_pairs, max_length=1024)
        return reranked_docs


if __name__ == '__main__':
    query = "如何在Python中创建一个列表"
    documents = [
        {"content": "Python列表是一个有序的集合，可以容纳任意数量的对象。"}
    ]

    reranker = LocalRerankModel()
    reranked_docs = reranker.rerank(query, documents, 10)
    for doc in reranked_docs:
        print(doc)

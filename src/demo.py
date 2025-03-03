import os.path

import dotenv

from src.application import Application
from src.model.embedding import OpenAILikeEmbeddingModel
from src.model.llm import LLMClient
from src.model.reranker import RerankAPIModel

dotenv.load_dotenv()


def main():
    app = Application(llm_client=LLMClient(base_url=os.getenv("OPENAI_API_BASE"), api_key=os.getenv("OPENAI_API_KEY")),
                      model="deepseek-chat",
                      embedding_model=OpenAILikeEmbeddingModel(),
                      rerank_model=RerankAPIModel())

    project_path = os.path.expanduser("~/workspace/spring-ai")
    project_name = project_path.split("/")[-1]
    app.index_project(project_path)
    while True:
        query = input("\nInput: ").strip().lower()
        if query == "exit":
            break

        try:
            print("\nAI Assistant:", end="", flush=True)
            for chunk in app.query(project_name, query):
                if chunk.choices[0].delta.content is not None:
                    print(chunk.choices[0].delta.content, end="", flush=True)
            print()  # New line after response
        except Exception as e:
            print(f"\nError processing query: {e}")


if __name__ == "__main__":
    main()

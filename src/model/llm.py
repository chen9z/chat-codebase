import os

import dotenv
from openai import OpenAI

dotenv.load_dotenv()


class LLMClient:
    def __init__(
            self,
            base_url="http://localhost:11434/v1",
            api_key="ollama",
            default_model="qwen2.5:7b-instruct-q6_K",
    ):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.default_model = default_model

    def get_response(self, messages, model=None, temperature=0.1, stream=False):
        try:
            model = model or self.default_model
            response = self.client.chat.completions.create(
                model=model, temperature=temperature, messages=messages, stream=stream
            )

            if not stream:
                return response.choices[0].message.content

            # 流式响应处理
            for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            print(f"Error in LLMClient: {str(e)}")
            return None


if __name__ == "__main__":
    # 测试本地模型
    local_client = LLMClient(base_url=os.getenv("OPENAI_API_BASE"), api_key=os.getenv("OPENAI_API_KEY"))
    # print(
    #     "Local model response:",
    #     local_client.get_response(model="deepseek-chat", messages=[{"role": "user", "content": "你是谁？"}]),
    # )

    # 测试流式响应
    print("\n=== Testing streaming response ===")
    print("Streaming response:", end=" ", flush=True)
    for chunk in local_client.get_response(
            model="deepseek-chat",
            messages=[{"role": "user", "content": "用简短的话介绍下你自己"}],
            stream=True
    ):
        print(chunk, end="", flush=True)
    print()  # 打印换行

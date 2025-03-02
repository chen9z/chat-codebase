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
        """
        初始化 LLM 客户端
        :param base_url: API 基础URL
        :param api_key: API密钥
        :param default_model: 默认使用的模型
        """
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
            content = []
            for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    content.append(chunk.choices[0].delta.content)
            return "".join(content)

        except Exception as e:
            print(f"Error in LLMClient: {str(e)}")
            return None


if __name__ == "__main__":
    # 测试本地模型
    local_client = LLMClient()
    print(
        "Local model response:",
        local_client.get_response([{"role": "user", "content": "你是谁？"}]),
    )

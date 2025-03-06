import json
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
            else:
                return self._stream_response(response)
        except Exception as e:
            print(f"Error in LLMClient: {str(e)}")
            return None

    def _stream_response(self, response):
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content

    def get_response_with_tools(self, messages, tools, model=None, temperature=0.1):
        try:
            model = model or self.default_model
            response = self.client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                stream=False
            )

            message = response.choices[0].message

            # 如果返回tool_calls,需要执行工具调用
            if message.tool_calls:
                return {
                    "type": "tool_calls",
                    "tool_calls": message.tool_calls
                }

            # 否则返回普通文本响应
            return {
                "type": "message",
                "content": message.content
            }

        except Exception as e:
            print(f"Error in LLMClient tool call: {str(e)}")
            print(json.dumps(messages, indent=2))
            return None


if __name__ == "__main__":
    local_client = LLMClient(base_url=os.getenv("OPENAI_API_BASE"), api_key=os.getenv("OPENAI_API_KEY"))

    # 测试普通响应
    print("\n=== Testing normal response ===")
    print(
        local_client.get_response(model="deepseek-chat", messages=[{"role": "user", "content": "你是谁？"}],
                                  stream=False)
    )

    # 测试流式响应
    print("\n=== Testing streaming response ===")
    stream = local_client.get_response(model="deepseek-chat", messages=[{"role": "user", "content": "你是谁？"}],
                                       stream=True)
    if stream:
        for chunk in stream:
            print(chunk, end='', flush=True)
    print("\n")

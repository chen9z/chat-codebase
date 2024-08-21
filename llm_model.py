import os

from openai import OpenAI
import dotenv

dotenv.load_dotenv()
openai = OpenAI(base_url=os.getenv("OPENAI_API_BASE"), api_key=os.getenv("OPENAI_API_KEY"))
ollama = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

def get_response_message_with_ollama(prompt: str, model="llama3:8b-instruct-q6_K", temperature=0.1) -> str:
    response = ollama.chat.completions.create(model=model,
                                              temperature=temperature,
                                              messages=[{"role": "user", "content": prompt}])
    return response.choices[0].message.content


def get_response(messages, model="llama3:8b-instruct-q6_K", temperature=0.1):
    response = ollama.chat.completions.create(model=model,
                                              temperature=temperature,
                                              messages=messages)
    return response.choices[0].message.content


def get_response_openai_like(messages, model="deepseek-coder", temperature=0.1):
    response = openai.chat.completions.create(model=model,
                                              temperature=temperature,
                                              messages=messages)
    return response.choices[0].message.contentt

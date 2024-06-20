from openai import OpenAI


ollama = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")


def get_response_message_with_ollama(prompt: str, model="llama3:8b-instruct-q6_K", temperature=0.1) -> str:
    response = ollama.chat.completions.create(model=model,
                                              temperature=temperature,
                                              messages=[{"role": "user", "content": prompt}])
    return response.choices[0].message.content




from llm_model import get_response
from repository import get_index

system_template = """
You are a large language AI assistant. You are given a user question, and please write clean, concise and accurate answer to the question.
Your answer must be correct, accurate and written by an expert using an unbiased and professional tone. Please limit to 1024 tokens.
Do not give any information that is not related to the question, and do not repeat. 
your answer must be written with **Chinese**
"""

user_template = """
Here are the set of contexts:

{context}

Remember, don't blindly repeat the contexts verbatim.
###
And here is the user question:
{question}

###
Answer:
"""

if __name__ == '__main__':
    project_path = "~/workspace/spring-ai"
    project_name = project_path.split("/")[-1]

    index = get_index()
    index.encode(project_path)
    result_ch = index.query_documents(project_name, "spring ai 项目是什么")

    while True:
        prompt = input("请输入问题：")
        if prompt == "exit":
            break
        documents = index.query_documents(project_name, prompt, limit=20)
        context = ""
        for doc in documents:
            context += f"file:///{doc.path} \n" + doc.content

        response = get_response(model="qwen2:7b-instruct-q6_K",
                                messages=[{"role": "system", "content": system_template},
                                          {"role": "user", "content": user_template.format(context=context,
                                                                                           question=prompt)}])
        print(f"AI Assistant:{response}")

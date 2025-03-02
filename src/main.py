import os.path

from src.application import Application


def main():
    app = Application()

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

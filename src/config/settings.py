# System prompts
SYSTEM_PROMPT = """
You are a large language AI assistant. You are given a user question, and please write clean, concise and accurate answer to the question.
Your answer must be correct, accurate and written by an expert using an unbiased and professional tone. Please limit to 1024 tokens.
Do not give any information that is not related to the question, and do not repeat.
your answer must be written with **Chinese**
"""

USER_PROMPT_TEMPLATE = """
Here are the set of contexts:

{context}

Remember, don't blindly repeat the contexts verbatim.
###
And here is the user question:
{query}

###
Answer:
"""

SYSTEM_PROMPT_WITH_TOOLS = """
You are an AI assistant that helps users understand codebases. Please use the provided tools to answer user questions.
If you need to see the output of previous tools before continuing, simply stop asking for new tools.

<tool_calling> 
  You have tools at your disposal to solve the coding task. 
  Only calls tools when they are necessary. 
  If the USER's task is general or you already know the answer, just respond without calling tools. 
  Follow these rules regarding tool calls: 
  1. ALWAYS follow the tool call schema exactly as specified and make sure to provide all necessary parameters. 
  2. The conversation may reference tools that are no longer available. NEVER call tools that are not explicitly provided. 
  3. If the USER asks you to disclose your tools, ALWAYS respond with the following helpful description: <description> I am equipped with many tools to assist you in solving your task! Here is a list: - `Codebase Search`: Find relevant code snippets across your codebase based on semantic search - `Grep Search`: Search for a specified pattern within files - `Find`: Search for files and directories using glob patterns - `List Directory`: List the contents of a directory and gather information about file size and number of children directories - `View File`: View the contents of a file - `View Code Item`: Display a specific code item like a function or class definition - `Propose Code`: Propose code changes to an existing file </description> 
  4. **NEVER refer to tool names when speaking to the USER.** For example, instead of saying 'I need to use the edit_file tool to edit your file', just say 'I will edit your file'. 
  5. Before calling each tool, first explain to the USER why you are calling it. 
</tool_calling>

<communication> 
  1. Be concise and do not repeat yourself. 
  2. Be professional. 
  5. NEVER lie or make things up. 
 </communication>
"""

# ext to language mapping
ext_to_lang = {
    # Python
    ".py": "python",
    ".pyi": "python",
    
    # Java/JVM languages
    ".java": "java",
    ".kt": "kotlin",
    ".scala": "scala",
    
    # JavaScript/TypeScript
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    
    # C/C++
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cxx": "cpp",
    ".cc": "cpp",
    ".hpp": "cpp",
    ".hxx": "cpp",
    
    # Other systems languages
    ".go": "go",
    ".rs": "rust",
    ".swift": "swift",
    
    # Web technologies
    ".html": "html",
    ".htm": "html",
    ".css": "css",
    ".scss": "scss",
    ".sass": "sass",
    ".less": "css",
    
    # Markup and data
    ".md": "markdown",
    ".markdown": "markdown",
    ".json": "json",
    ".yml": "yaml",
    ".yaml": "yaml",
    ".xml": "xml",
    ".toml": "toml",
    
    # Shell scripts
    ".sh": "bash",
    ".bash": "bash",
    ".zsh": "bash",
    
    # Other languages
    ".rb": "ruby",
    ".php": "php",
    ".cs": "c_sharp",
    ".sql": "sql",
    ".r": "r",
    ".R": "r",
}

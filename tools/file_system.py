import os
from langchain.tools import tool

# This dictates where the data physically lives, just like the FileSystemBackend
ROOT_DIRECTORY = "./projects"
os.makedirs(ROOT_DIRECTORY, exist_ok=True)

@tool
def save_file_to_disk(filename: str, content: str) -> str:
    """
    Saves text content, code, or research to a permanent file on the local hard drive.
    Args:
        filename: The name of the file (e.g., 'report.md', 'script.py').
        content: The actual text or code to save inside the file.
    """
    # Security: os.path.basename prevents the AI from writing to system folders like C:/Windows
    safe_filename = os.path.basename(filename)
    file_path = os.path.join(ROOT_DIRECTORY, safe_filename)
    
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Success! File permanently saved to the hard drive at: {file_path}"
    except Exception as e:
        return f"Error saving file: {str(e)}"
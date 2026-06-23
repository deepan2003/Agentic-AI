import os
from langchain.tools import tool
from e2b_code_interpreter import Sandbox
from dotenv import load_dotenv

load_dotenv()

@tool
def execute_python_code(code: str) -> str:
    """
    Execute Python code in a secure E2B cloud sandbox. 
    Use this to perform math, data analysis, write algorithms, or test logic.
    Args:
        code: The Python code snippet to run.
    """
    print("\n[Sandbox] Booting up secure E2B Cloud Environment...")
    
    # Check if the API key is set
    if not os.getenv("E2B_API_KEY"):
        return "Error: E2B_API_KEY is missing from the .env file."

    try:
        # Create an ephemeral, secure sandbox session
        with Sandbox.create() as sandbox:
            print("[Sandbox] Executing AI-generated code...")
            execution = sandbox.run_code(code)
            
            # Catch Python execution errors (e.g., syntax errors, missing variables)
            if execution.error:
                return f"Error executing code: {execution.error.name} - {execution.error.value}"
            
            # Gather standard print() outputs and cell results
            results = []
            if execution.logs.stdout:
                results.append("\n".join(execution.logs.stdout))
            if execution.results:
                for result in execution.results:
                    if result.is_main_result:
                        results.append(str(result.text))
            
            return "\n".join(results) if results else "Code executed successfully with no output."
            
    except Exception as e:
        return f"Sandbox connection failed: {str(e)}"
import os
import re
from langchain.tools import tool

ROOT_DIRECTORY = "./projects"
os.makedirs(ROOT_DIRECTORY, exist_ok=True)

def clean_document_content(content: str) -> str:
    """
    Cleans content by removing Python script boilerplate or code blocks
    if the AI mistakenly output Python code instead of plain text.
    """
    # If the entire content is wrapped in a python code block, check if it contains hardcoded text
    if "```python" in content:
        # Check if the code was a script to create a document with headlines
        headlines = re.findall(r'["\']headline["\']\s*:\s*["\']([^"\']+)["\']', content)
        points = re.findall(r'["\']points["\']\s*:\s*["\']([^"\']+)["\']', content)
        if headlines:
            cleaned = []
            for h, p in zip(headlines, points if len(points) == len(headlines) else [""] * len(headlines)):
                cleaned.append(f"## {h}")
                if p:
                    cleaned.append(f"{p}\n")
            return "\n".join(cleaned)
        
        # Otherwise remove the code block wrappers
        content = re.sub(r'```python[\s\S]*?```', '', content).strip()
        
    return content.strip()

@tool
def save_file_to_disk(filename: str, content: str) -> str:
    """
    Saves text content, code, or research to a permanent file on the local hard drive.
    For .docx/.doc files, creates a genuine formatted Microsoft Word document.
    Args:
        filename: The name of the file (e.g., 'report.docx', 'news.txt').
        content: The actual text or research content to save inside the file.
    """
    safe_filename = os.path.basename(filename)
    file_path = os.path.join(ROOT_DIRECTORY, safe_filename)
    clean_text = clean_document_content(content)

    try:
        # Handle Microsoft Word documents properly
        if safe_filename.lower().endswith((".docx", ".doc")):
            try:
                from docx import Document
                from docx.shared import Pt, Inches
                
                doc = Document()
                
                # Document title
                title = os.path.splitext(safe_filename)[0].replace("_", " ").title()
                doc.add_heading(title, level=1)
                
                # Add formatted paragraphs
                for line in clean_text.splitlines():
                    stripped = line.strip()
                    if not stripped:
                        continue
                    if stripped.startswith("# "):
                        doc.add_heading(stripped[2:], level=1)
                    elif stripped.startswith("## "):
                        doc.add_heading(stripped[3:], level=2)
                    elif stripped.startswith("### "):
                        doc.add_heading(stripped[4:], level=3)
                    elif stripped.startswith(("- ", "* ", "• ")):
                        p = doc.add_paragraph(stripped[2:], style='List Bullet')
                        p.paragraph_format.space_after = Pt(4)
                    else:
                        p = doc.add_paragraph(stripped)
                        p.paragraph_format.space_after = Pt(6)
                        
                doc.save(file_path)
                return f"Success! Real Microsoft Word document (.docx) permanently saved to: {file_path}"
                
            except ImportError:
                # Fallback if python-docx isn't available
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(clean_text)
                return f"Success! Document saved as text to: {file_path}"
        else:
            # Handle plain text / markdown / code files
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(clean_text)
            return f"Success! File permanently saved to the hard drive at: {file_path}"
            
    except Exception as e:
        return f"Error saving file: {str(e)}"

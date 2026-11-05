import os
import re
from langchain.tools import tool

ROOT_DIRECTORY = "./projects"
os.makedirs(ROOT_DIRECTORY, exist_ok=True)


def sanitize_user_folder(thread_id: str = "default") -> str:
    """Returns a safe, lowercase alphanumeric user folder name for strict per-user isolation."""
    if not thread_id:
        return "default"
    clean = re.sub(r'[^a-zA-Z0-9_\-]', '_', str(thread_id).strip().lower()).strip('_')
    return clean or "default"


def get_user_projects_dir(thread_id: str = "default") -> str:
    """Returns the strictly isolated saved-documents folder (`./projects/<user>/`) for this user."""
    safe_user = sanitize_user_folder(thread_id)
    user_dir = os.path.join(ROOT_DIRECTORY, safe_user)
    os.makedirs(user_dir, exist_ok=True)
    return user_dir


def clean_document_content(content: str) -> str:
    """
    Cleans content by removing Python script boilerplate or code blocks
    if the AI mistakenly output Python code instead of plain text.
    """
    if not content:
        return ""

    # If the entire content is wrapped in a python code block, check if it contains hardcoded text
    if "```python" in content:
        headlines = re.findall(r'["\']headline["\']\s*:\s*["\']([^"\']+)["\']', content)
        points = re.findall(r'["\']points["\']\s*:\s*["\']([^"\']+)["\']', content)
        if headlines:
            cleaned = []
            for h, p in zip(headlines, points if len(points) == len(headlines) else [""] * len(headlines)):
                cleaned.append(f"## {h}")
                if p:
                    cleaned.append(f"{p}\n")
            return "\n".join(cleaned)

        content = re.sub(r'```python[\s\S]*?```', '', content).strip()

    return content.strip()


def _add_formatted_runs(paragraph, text: str):
    """Parses inline Markdown bold (**text**) into proper python-docx bold runs."""
    parts = re.split(r'(\*\*.*?\*\*)', text)
    for part in parts:
        if part.startswith("**") and part.endswith("**") and len(part) > 4:
            run = paragraph.add_run(part[2:-2])
            run.bold = True
        else:
            paragraph.add_run(part)


@tool
def save_file_to_disk(filename: str, content: str, thread_id: str = "default") -> str:
    """
    Saves text content, code, or research to a permanent file in the user's private folder (`./projects/<thread_id>/`).
    For .docx/.doc files, creates a genuine formatted Microsoft Word document.
    Args:
        filename: The name of the file (e.g., 'karur_news.docx', 'news.txt').
        content: The actual text or research content to save inside the file.
        thread_id: The user identifier to isolate saved files per user.
    """
    safe_filename = os.path.basename(filename)
    if safe_filename.lower().endswith(".doc"):
        safe_filename = safe_filename[:-4] + ".docx"

    user_dir = get_user_projects_dir(thread_id)
    file_path = os.path.join(user_dir, safe_filename)
    clean_text = clean_document_content(content)

    try:
        # Handle Microsoft Word documents properly
        if safe_filename.lower().endswith(".docx"):
            try:
                from docx import Document
                from docx.shared import Pt

                doc = Document()

                # Document title
                title = os.path.splitext(safe_filename)[0].replace("_", " ").title()
                doc.add_heading(title, level=1)

                # Add formatted paragraphs with clean Markdown-to-Word conversion
                for line in clean_text.splitlines():
                    stripped = line.strip()
                    if not stripped:
                        continue
                    if stripped.startswith("# "):
                        doc.add_heading(stripped[2:].replace("**", ""), level=1)
                    elif stripped.startswith("## "):
                        doc.add_heading(stripped[3:].replace("**", ""), level=2)
                    elif stripped.startswith("### "):
                        doc.add_heading(stripped[4:].replace("**", ""), level=3)
                    elif stripped.startswith(("- ", "* ", "• ")):
                        p = doc.add_paragraph(style='List Bullet')
                        _add_formatted_runs(p, stripped[2:].strip())
                        p.paragraph_format.space_after = Pt(4)
                    else:
                        p = doc.add_paragraph()
                        _add_formatted_runs(p, stripped)
                        p.paragraph_format.space_after = Pt(6)

                doc.save(file_path)
                return f"Success! Real Microsoft Word document (.docx) permanently saved to: {file_path}"

            except ImportError:
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
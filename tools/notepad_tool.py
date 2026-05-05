import subprocess
import sys
import os
from typing import Optional


def open_notepad(filename: Optional[str] = None) -> str:
    """
    Open a text editor (notepad/TextEdit) for writing.
    
    Args:
        filename: Optional filename to open or create. If not provided, opens a blank notepad.
    
    Returns:
        A message indicating the notepad was opened.
    """
    try:
        if filename:
            file_path = os.path.abspath(filename)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            if not os.path.exists(file_path):
                with open(file_path, "a", encoding="utf-8"):
                    pass
        else:
            file_path = None

        if sys.platform == "darwin":
            # macOS: use TextEdit
            if file_path:
                subprocess.Popen(["open", "-a", "TextEdit", file_path])
                return f"Opened TextEdit with file: {file_path}"
            else:
                subprocess.Popen(["open", "-a", "TextEdit"])
                return "Opened TextEdit"
        elif sys.platform == "win32":
            # Windows: use notepad.exe
            if file_path:
                subprocess.Popen(["notepad.exe", file_path])
                return f"Opened notepad with file: {file_path}"
            else:
                subprocess.Popen(["notepad.exe"])
                return "Opened notepad"
        else:
            # Linux: try gedit or nano
            if file_path:
                subprocess.Popen(["gedit", file_path])
                return f"Opened text editor with file: {file_path}"
            else:
                subprocess.Popen(["gedit"])
                return "Opened text editor"
    except Exception as e:
        return f"Failed to open text editor: {str(e)}"

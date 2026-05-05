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
        if sys.platform == "darwin":
            # macOS: use TextEdit
            if filename:
                subprocess.Popen(["open", "-a", "TextEdit", filename])
                return f"Opened TextEdit with file: {filename}"
            else:
                subprocess.Popen(["open", "-a", "TextEdit"])
                return "Opened TextEdit"
        elif sys.platform == "win32":
            # Windows: use notepad.exe
            if filename:
                subprocess.Popen(["notepad.exe", filename])
                return f"Opened notepad with file: {filename}"
            else:
                subprocess.Popen(["notepad.exe"])
                return "Opened notepad"
        else:
            # Linux: try gedit or nano
            if filename:
                subprocess.Popen(["gedit", filename])
                return f"Opened text editor with file: {filename}"
            else:
                subprocess.Popen(["gedit"])
                return "Opened text editor"
    except Exception as e:
        return f"Failed to open text editor: {str(e)}"

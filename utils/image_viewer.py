"""
image_viewer.py
-----------------
Opens a saved visualization PNG using the operating system's default image
viewer. Works on Windows, macOS and Linux, and never crashes the GUI if
the image is missing or the OS refuses to open it.
"""

import os
import platform
import subprocess
from tkinter import messagebox


def open_image(path):
    if not os.path.exists(path):
        messagebox.showerror(
            "Image Not Found",
            f"Could not find the visualization file:\n{path}\n\n"
            "Try restarting the application so it can be regenerated.",
        )
        return

    try:
        system = platform.system()
        if system == "Windows":
            os.startfile(path)  # noqa: this attribute only exists on Windows
        elif system == "Darwin":
            subprocess.run(["open", path], check=False)
        else:
            subprocess.run(["xdg-open", path], check=False)
    except Exception as exc:
        messagebox.showerror(
            "Could Not Open Image",
            f"Please open this file manually:\n{path}\n\nError: {exc}",
        )

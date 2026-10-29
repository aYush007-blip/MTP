#!/usr/bin/env python3
"""
Entrypoint for Jetson Speech-to-LaTeX Showcase GUI.
Allows running `python gui.py` directly from main directory.
"""
import os
import sys
from pathlib import Path

# Auto-bootstrap to local .venv if running under system python
venv_python = Path(__file__).resolve().parent / ".venv" / "bin" / "python"
if venv_python.exists() and sys.executable != str(venv_python):
    try:
        import aiohttp
    except ImportError:
        os.execv(str(venv_python), [str(venv_python)] + sys.argv)

# Add main, src, and gui to sys.path
MAIN_DIR = Path(__file__).resolve().parent
if str(MAIN_DIR) not in sys.path:
    sys.path.insert(0, str(MAIN_DIR))
if str(MAIN_DIR / "src") not in sys.path:
    sys.path.insert(0, str(MAIN_DIR / "src"))
if str(MAIN_DIR / "gui") not in sys.path:
    sys.path.insert(0, str(MAIN_DIR / "gui"))

from gui.server import create_app, ensure_edge_daemons, get_slm_translator
from aiohttp import web

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print("\n=======================================================")
    print("[GUI Server] Jetson Orin Speech-to-LaTeX Showcase GUI")
    print(f"[GUI Server] Local Server URL: http://0.0.0.0:{port}")
    print("=======================================================\n")
    ensure_edge_daemons()
    get_slm_translator()
    app = create_app()
    web.run_app(app, host="0.0.0.0", port=port)

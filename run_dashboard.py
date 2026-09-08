"""
Launcher for AURA EDTECH AI Student Performance & Learning Dashboard.
Starts the FastAPI backend and serves the luxury web frontend.
"""
import sys
import os
import webbrowser
import threading
import time
import uvicorn

# Configure UTF-8 for Windows consoles
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def open_browser():
    time.sleep(1.2)
    url = "http://127.0.0.1:8000/dashboard/index.html"
    print("\n" + "=" * 55)
    print(">> AURA EDTECH AI Learning Dashboard is Live!")
    print(f">> URL: {url}")
    print("=" * 55 + "\n")
    try:
        webbrowser.open(url)
    except Exception:
        pass

if __name__ == "__main__":
    threading.Thread(target=open_browser, daemon=True).start()
    uvicorn.run("src.api:app", host="127.0.0.1", port=8000, log_level="info")

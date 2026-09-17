"""
Main Entry Point for Student Performance Prediction System & Vercel Serverless.
"""

import sys
import os

# Expose top-level FastAPI instance for Vercel, Uvicorn, and ASGI cloud servers
from src.api import app

def print_banner():
    print("""
===================================================================
🎓 STUDENT PERFORMANCE PREDICTION SYSTEM
Predict Student Performance using Feature Selection & Classification
===================================================================
Available modes:
  1. Train Baseline Models:   python cli.py train --fs_method consensus --k 8
  2. Launch Web Dashboard:    python run_dashboard.py
  3. Launch Streamlit Hub:    streamlit run streamlit_app.py
  4. Run Pytest Test Suite:    pytest tests/ -v
===================================================================
""")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] != "run":
        import cli
        cli.main()
    else:
        print_banner()
        import uvicorn
        port = int(os.environ.get("PORT", 8000))
        uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)

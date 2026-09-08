"""
Main Entry Point for Student Performance Prediction System.
"""

import sys
import os

def print_banner():
    print("""
===================================================================
🎓 STUDENT PERFORMANCE PREDICTION SYSTEM
Predict Student Performance using Feature Selection & Classification
===================================================================
Available modes:
  1. Train Baseline Models:  python cli.py train --fs_method consensus --k 8
  2. Launch Web Dashboard:    streamlit run app.py
  3. Launch FastAPI Backend:  uvicorn src.api:app --reload --port 8000
  4. Run Pytest Test Suite:   pytest tests/ -v
===================================================================
""")

if __name__ == "__main__":
    print_banner()
    if len(sys.argv) > 1:
        import cli
        cli.main()
    else:
        print("Tip: Run 'streamlit run app.py' to open the interactive web UI.")

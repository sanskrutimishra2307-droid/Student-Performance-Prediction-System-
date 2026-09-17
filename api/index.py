import sys
import os

# Add root directory to Python module search path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from src.api import app

# Export app for Vercel Serverless Function runtime
__all__ = ["app"]

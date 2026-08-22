"""
pytest configuration — adds the project root to sys.path so that
all `src.*` imports work without installation.
"""
import sys
from pathlib import Path

# Insert project root so imports resolve
sys.path.insert(0, str(Path(__file__).resolve().parent))

"""
Aleth — OpenEnv Server Entry Point (server/app.py)

This is the multi-mode deployment entry point required by openenv validate.
It exposes the AlethEnv over HTTP on port 7860.

Usage:
    uv run --project . server
    uvicorn server.app:app --host 0.0.0.0 --port 7860
"""

import sys
import os

# Ensure the repo root is on sys.path so flat-structure imports work
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

# Import the fully-compliant FastAPI app from root main.py
from main import app  # noqa: E402


def main():
    """
    Entry point for: uv run --project . server
    """
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()

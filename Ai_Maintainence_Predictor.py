"""
Backward-compatibility alias for the Streamlit web dashboard.
Delegates to the canonical entrypoint in app.py.
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app import main

if __name__ == "__main__":
    main()

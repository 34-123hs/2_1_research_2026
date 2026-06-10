"""Make the repo root importable from tests (so `import model` etc. works)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

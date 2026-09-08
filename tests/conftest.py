import os
import sys

# Make the repository root importable so `import main` works regardless of
# the directory pytest is invoked from.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

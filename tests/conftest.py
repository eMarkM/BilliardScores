import sys
from pathlib import Path

# Ensure src/ is importable in tests (src layout)
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Ensure project root is also importable for integration tests that import scripts.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

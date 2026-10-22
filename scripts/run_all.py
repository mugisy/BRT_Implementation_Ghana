"""Convenience wrapper: `python scripts/run_all.py --steps all`."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from brt.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
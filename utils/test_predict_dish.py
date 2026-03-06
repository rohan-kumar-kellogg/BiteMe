import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from models.retrieval import main as retrieval_main


if __name__ == "__main__":
    # Thin wrapper to keep CLI discoverable under utils/.
    parser = argparse.ArgumentParser(add_help=False)
    parser.parse_known_args()
    retrieval_main()


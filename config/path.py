from pathlib import Path

# Project root: .../project
PROJECT_ROOT = Path(__file__).resolve().parent.parent

VIETNAMESE_STOPWORDS = PROJECT_ROOT / "stopwords" / "vietnamese-stopwords.txt"
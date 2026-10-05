from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"

IMAGE_PATH = DATA_DIR / "puzzle2.png"
TEMPLATE_DIR = DATA_DIR / "templates"
SOLUTION_IMAGE_PATH = DATA_DIR / f"{IMAGE_PATH.stem}_solved.png"
GRID_SIZE = 6

VICTIM = "Josh"
BLOCKED_OBJECTS = ("shelf", "plant", "table")
JSON_INDENT = 2

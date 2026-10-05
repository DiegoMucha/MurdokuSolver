# MurdokuSolver

MurdokuSolver reads a Murdoku puzzle image and determines the murderer. The
application combines three stages:

## Requirements

- Python 3.12 or newer
- Tesseract OCR installed on the operating system
- The Python packages listed in `requirements.txt`

Tesseract is not a Python package and must be installed separately. On
Ubuntu/Debian:

```bash
sudo apt update
sudo apt install tesseract-ocr
```

On macOS with Homebrew:

```bash
brew install tesseract
```

Confirm that Tesseract is available:

```bash
tesseract --version
```

## Installation

Clone or download the project, enter its directory, and create a virtual
environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows, activate the environment with:

```powershell
.venv\Scripts\activate
```

The project can alternatively be installed with `uv`:

```bash
uv sync
```

## Configuration

Runtime settings are in `config.py`. The important options are:

- `IMAGE_PATH`: puzzle image to process.
- `TEMPLATE_DIR`: directory containing the board object templates.
- `GRID_SIZE`: number of rows and columns.
- `VICTIM`: name of the victim.
- `BLOCKED_OBJECTS`: objects on which a person cannot stand.

For example, to process the second included puzzle, set:

```python
IMAGE_PATH = DATA_DIR / "puzzle2.png"
```

## Execution

Run the complete pipeline from the project directory:

```bash
python main.py
```

When using the included virtual environment without activating it:

```bash
.venv/bin/python main.py
```

The program prints a combined JSON document containing:

- The raw board-reader result
- The raw OCR text
- The parsed OCR constraints
- The CP-SAT solution and detected positions

The final terminal line reports the result directly:

```text
The killer is: Casey
```

If the constraints have no valid solution, the program prints:

```text
No solution was found.
```

## Project structure

```text
MurdokuSolver/
├── config.py             # Paths and puzzle settings
├── main.py               # Complete application pipeline
├── cv/
│   ├── board_reader.py   # Board, room, and object recognition
│   └── ocr.py            # OCR and natural-language clue parsing
├── solver/
│   └── solver.py         # OR-Tools CP-SAT model
├── data/
│   ├── puzzle1.png
│   ├── puzzle2.png
│   └── templates/        # Object templates used by OpenCV
└── requirements.txt
```

## Using the modules directly

The complete pipeline is available as a function:

```python
from main import run_pipeline

result = run_pipeline()
print(result["solver"]["murderer"])
```

The vision outputs can also be passed directly to the solver package:

```python
from solver import solve_vision_outputs

solution = solve_vision_outputs(board_output, parsed_ocr_output)
```

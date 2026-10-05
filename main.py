import json

from config import (
    BLOCKED_OBJECTS,
    GRID_SIZE,
    IMAGE_PATH,
    JSON_INDENT,
    TEMPLATE_DIR,
    VICTIM,
)
from cv import analyze_murdoku, parse_clues, read_clues
from solver import solve_vision_outputs


def run_pipeline():
    """Run board reading, OCR parsing, and constraint solving."""
    board_reader_output = analyze_murdoku(
        image_path=IMAGE_PATH,
        template_dir=TEMPLATE_DIR,
        grid_size=GRID_SIZE,
    )
    raw_ocr_output = read_clues(str(IMAGE_PATH))
    parsed_ocr_output = parse_clues(raw_ocr_output)
    solver_output = solve_vision_outputs(
        board_reader_output,
        parsed_ocr_output,
        victim=VICTIM,
        blocked_objects=BLOCKED_OBJECTS,
    )

    return {
        "board_reader": board_reader_output,
        "ocr_raw": raw_ocr_output,
        "ocr_parsed": parsed_ocr_output,
        "solver": solver_output,
    }


def main():
    output = run_pipeline()
    print(json.dumps(output, ensure_ascii=False, indent=JSON_INDENT))

    solver_output = output["solver"]
    if solver_output is None:
        print("\nNo solution was found.")
    else:
        print(f"\nThe killer is: {solver_output['murderer']}")


if __name__ == "__main__":
    main()

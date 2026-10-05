import json

from config import (
    BLOCKED_OBJECTS,
    GRID_SIZE,
    IMAGE_PATH,
    JSON_INDENT,
    SOLUTION_IMAGE_PATH,
    TEMPLATE_DIR,
    VICTIM,
)
from cv import analyze_murdoku, parse_clues, read_clues, render_solution
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
    puzzle_name = IMAGE_PATH.stem
    print(f"Solving {puzzle_name}...")

    output = run_pipeline()
    # print(json.dumps(output, ensure_ascii=False, indent=JSON_INDENT))

    solver_output = output["solver"]
    if solver_output is None:
        print("Puzzle could not be solved.")
    else:
        solution_image = render_solution(
            IMAGE_PATH,
            solver_output,
            output_path=SOLUTION_IMAGE_PATH,
            grid_size=GRID_SIZE,
            victim=VICTIM,
        )
        print(f"Puzzle Solved! The killer is: {solver_output['murderer']}")
        print(f"Solution image saved to: {solution_image}")


if __name__ == "__main__":
    main()

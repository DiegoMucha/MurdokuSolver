"""Render a solved Murdoku result on top of its detected board."""

from pathlib import Path

import cv2
import numpy as np

from .board_reader import detect_board


def _fit_font_scale(text, cell_width, font, thickness):
    """Choose a font scale that keeps a name inside one grid cell."""
    scale = max(cell_width / 150, 0.35)
    maximum_width = cell_width * 0.86

    while scale > 0.25:
        (text_width, _), _ = cv2.getTextSize(text, font, scale, thickness)
        if text_width <= maximum_width:
            return scale
        scale -= 0.05

    return 0.25


def render_solution(
    image_path,
    solver_output,
    *,
    output_path=None,
    grid_size=6,
    victim=None,
):
    """Draw all solved positions on the board and return the saved image path.

    Every person is drawn at the center of their solved cell. The killer's name
    is red, the victim's name is green, and all other names are black. Empty
    cells receive a large black X. Rows and columns in ``solver_output`` use the
    solver's zero-based coordinates.
    """
    if solver_output is None:
        raise ValueError("cannot render a puzzle without a solver result")

    image_path = Path(image_path)
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"image not found or unreadable: {image_path}")

    positions = solver_output.get("positions")
    killer = solver_output.get("murderer")
    if not isinstance(positions, dict) or not killer:
        raise ValueError("solver output must contain positions and murderer")

    board = detect_board(image)
    board_height, board_width = board.shape[:2]
    x_edges = np.linspace(0, board_width, grid_size + 1).round().astype(int)
    y_edges = np.linspace(0, board_height, grid_size + 1).round().astype(int)

    font = cv2.FONT_HERSHEY_SIMPLEX
    text_thickness = max(1, round(min(board_width, board_height) / 500))
    outline_thickness = text_thickness + 3
    x_thickness = max(3, round(min(board_width, board_height) / 220))

    occupied_cells = {
        (int(position["row"]), int(position["column"]))
        for position in positions.values()
    }

    for row in range(grid_size):
        for column in range(grid_size):
            if (row, column) in occupied_cells:
                continue

            x1, x2 = x_edges[column], x_edges[column + 1]
            y1, y2 = y_edges[row], y_edges[row + 1]
            margin_x = max(10, round((x2 - x1) * 0.25))
            margin_y = max(10, round((y2 - y1) * 0.25))
            cv2.line(
                board,
                (x1 + margin_x, y1 + margin_y),
                (x2 - margin_x, y2 - margin_y),
                (0, 0, 0),
                x_thickness,
                cv2.LINE_AA,
            )
            cv2.line(
                board,
                (x2 - margin_x, y1 + margin_y),
                (x1 + margin_x, y2 - margin_y),
                (0, 0, 0),
                x_thickness,
                cv2.LINE_AA,
            )

    for person, position in positions.items():
        row = int(position["row"])
        column = int(position["column"])
        if not 0 <= row < grid_size or not 0 <= column < grid_size:
            raise ValueError(
                f"position for {person!r} is outside the {grid_size}x{grid_size} board"
            )

        x1, x2 = x_edges[column], x_edges[column + 1]
        y1, y2 = y_edges[row], y_edges[row + 1]
        cell_width = x2 - x1
        cell_height = y2 - y1
        font_scale = _fit_font_scale(person, cell_width, font, text_thickness)
        (text_width, text_height), _ = cv2.getTextSize(
            person,
            font,
            font_scale,
            text_thickness,
        )
        text_x = x1 + (cell_width - text_width) // 2
        text_y = y1 + (cell_height + text_height) // 2

        if person == killer:
            text_color = (0, 0, 255)
            outline_color = (0, 0, 0)
        elif person == victim:
            text_color = (0, 255, 0)
            outline_color = (0, 0, 0)
        else:
            text_color = (0, 0, 0)
            outline_color = (255, 255, 255)

        # The contrasting outline keeps labels readable over rooms and objects.
        cv2.putText(
            board,
            person,
            (text_x, text_y),
            font,
            font_scale,
            outline_color,
            outline_thickness,
            cv2.LINE_AA,
        )
        cv2.putText(
            board,
            person,
            (text_x, text_y),
            font,
            font_scale,
            text_color,
            text_thickness,
            cv2.LINE_AA,
        )

    if output_path is None:
        output_path = image_path.with_name(f"{image_path.stem}_solved.png")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not cv2.imwrite(str(output_path), board):
        raise OSError(f"could not save solution image: {output_path}")

    return output_path

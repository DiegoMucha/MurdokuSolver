import json
from pathlib import Path

import cv2
import numpy as np


ROOM_SUFFIXES = {
    "a":  {"room": "yellow", "room_id": 1},
    "az": {"room": "blue",   "room_id": 2},
    "v":  {"room": "green",  "room_id": 3},
    "n":  {"room": "orange", "room_id": 4},
}


def detect_board(image, min_area_ratio=0.25, dark_threshold=100):
    dark_mask = np.all(
        image < dark_threshold,
        axis=2
    ).astype(np.uint8) * 255

    contours, _ = cv2.findContours(
        dark_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    h_img, w_img = image.shape[:2]
    image_area = h_img * w_img

    candidates = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h

        if area / image_area >= min_area_ratio:
            candidates.append((area, x, y, w, h))

    if not candidates:
        return image.copy()

    candidates.sort(reverse=True)
    _, x, y, w, h = candidates[0]

    return image[y:y + h, x:x + w].copy()


def build_grid(board, grid_size):
    h, w = board.shape[:2]

    x_edges = np.linspace(
        0, w, grid_size + 1
    ).round().astype(int)

    y_edges = np.linspace(
        0, h, grid_size + 1
    ).round().astype(int)

    cells = []

    for row in range(grid_size):
        for col in range(grid_size):
            x1 = int(x_edges[col])
            x2 = int(x_edges[col + 1])
            y1 = int(y_edges[row])
            y2 = int(y_edges[row + 1])

            cells.append({
                "row": row + 1,
                "col": col + 1,
                "bbox": (x1, y1, x2, y2),
            })

    return cells


def load_templates(template_dir):
    template_dir = Path(template_dir)

    if not template_dir.exists():
        raise FileNotFoundError(
            f"Template directory not found: {template_dir}"
        )

    templates = {}

    for path in sorted(template_dir.iterdir()):
        if path.suffix.lower() not in {
            ".png", ".jpg", ".jpeg", ".webp"
        }:
            continue

        template = cv2.imread(
            str(path),
            cv2.IMREAD_COLOR
        )

        if template is None:
            continue

        templates[path.stem.lower()] = template

    if not templates:
        raise ValueError(
            f"No valid template images found in: {template_dir}"
        )

    return templates


def resize_template_to_cell(template, cell_img):
    h, w = cell_img.shape[:2]

    if (
        template.shape[0] >= h
        and template.shape[1] >= w
    ):
        interpolation = cv2.INTER_AREA
    else:
        interpolation = cv2.INTER_CUBIC

    return cv2.resize(
        template,
        (w, h),
        interpolation=interpolation,
    )


def compare_cell_to_template(cell_img, template):
    template_resized = resize_template_to_cell(
        template,
        cell_img,
    )

    result = cv2.matchTemplate(
        cell_img,
        template_resized,
        cv2.TM_SQDIFF_NORMED,
    )

    return float(result[0, 0])


def best_template_for_cell(cell_img, templates):
    best_name = None
    best_difference = float("inf")

    for template_name, template in templates.items():
        difference = compare_cell_to_template(
            cell_img,
            template,
        )

        if difference < best_difference:
            best_difference = difference
            best_name = template_name

    return best_name, best_difference


def parse_template_name(template_name):
    if not template_name or "_" not in template_name:
        raise ValueError(
            f"Invalid template name: {template_name!r}. "
            "Expected a name ending in _a, _az, _v or _n."
        )

    object_name, suffix = template_name.rsplit("_", 1)

    room_info = ROOM_SUFFIXES.get(suffix)

    if room_info is None:
        raise ValueError(
            f"Unknown room suffix '_{suffix}' in template "
            f"{template_name!r}."
        )

    return {
        "object": object_name,
        "room": room_info["room"],
        "room_id": room_info["room_id"],
    }


def analyze_murdoku(image_path, template_dir, grid_size=6, max_template_difference=None):
    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise FileNotFoundError(
            f"Image not found or unreadable: {image_path}"
        )

    templates = load_templates(template_dir)
    board = detect_board(image)
    cells = build_grid(board, grid_size)

    result_cells = []

    for cell in cells:
        x1, y1, x2, y2 = cell["bbox"]
        cell_img = board[y1:y2, x1:x2]

        template_name, difference = best_template_for_cell(
            cell_img,
            templates,
        )

        if (
            max_template_difference is not None
            and difference > max_template_difference
        ):
            result_cells.append({
                "row": cell["row"],
                "col": cell["col"],
                "room_id": 0,
                "room": "unknown",
                "object": None,
            })
            continue

        parsed = parse_template_name(template_name)

        result_cells.append({
            "row": cell["row"],
            "col": cell["col"],
            "room_id": parsed["room_id"],
            "room": parsed["room"],
            "object": parsed["object"],
        })

    return {
        "grid_size": grid_size,
        "cells": result_cells,
    }


def analyze_murdoku_json(image_path, template_dir, grid_size=6, max_template_difference=None, indent=2):
    result = analyze_murdoku(
        image_path=image_path,
        template_dir=template_dir,
        grid_size=grid_size,
        max_template_difference=max_template_difference,
    )

    return json.dumps(
        result,
        ensure_ascii=False,
        indent=indent,
    )
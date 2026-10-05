"""Constraint-programming solver for Murdoku puzzles.

The public adapter accepts the JSON-compatible dictionaries returned by
``cv.board_reader.analyze_murdoku`` and ``cv.ocr.parse_clues``.  Coordinates
are zero-based: row 0 is north and column 0 is west.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ortools.sat.python import cp_model


DEFAULT_VICTIM = "Josh"
DEFAULT_BLOCKED_OBJECTS = {"shelf", "plant", "table"}

OBJECT_ALIASES = {
    "alfombra": "carpet",
    "cama": "bed",
    "carpet": "carpet",
    "chair": "chair",
    "fondo": None,
    "libreria": "shelf",
    "mesa": "table",
    "plant": "plant",
    "planta": "plant",
    "shelf": "shelf",
    "silla": "chair",
    "slla": "chair",
    "table": "table",
}

CELL_RELATIONS = {
    "in room",
    "not in room",
    "on top",
    "not on top",
    "beside",
    "not beside",
    "row",
    "column",
}
PERSON_RELATIONS = {
    "same room",
    "different room",
    "alone with",
    "north of",
    "south of",
    "east of",
    "west of",
}
SUPPORTED_RELATIONS = CELL_RELATIONS | PERSON_RELATIONS | {"alone"}
DIRECTIONS = {
    "north of": ("row", -1),
    "south of": ("row", 1),
    "west of": ("column", -1),
    "east of": ("column", 1),
}


def _normalise_object(object_name: Any) -> str | None:
    if object_name is None:
        return None

    key = str(object_name).strip().lower()
    return OBJECT_ALIASES.get(key, key)


def _people_from_clues(clues: list[list[Any]], victim: str) -> list[str]:
    people = []

    def add(person: Any) -> None:
        if isinstance(person, str) and person and person not in people:
            people.append(person)

    for clue in clues:
        if clue:
            add(clue[0])

    add(victim)
    return people


def puzzle_from_vision(
    board_output: dict[str, Any],
    parsed_ocr_output: dict[str, Any],
    *,
    victim: str = DEFAULT_VICTIM,
    people: list[str] | None = None,
    blocked_objects: set[str] | list[str] | tuple[str, ...] = DEFAULT_BLOCKED_OBJECTS,
) -> dict[str, Any]:
    """Adapt board-reader and parsed-OCR output to the solver input format."""
    size = board_output.get("grid_size")
    cells = board_output.get("cells")
    clues = parsed_ocr_output.get("clues")

    if not isinstance(size, int) or size <= 0:
        raise ValueError("board output must contain a positive integer grid_size")
    if not isinstance(cells, list):
        raise ValueError("board output must contain a cells list")
    if not isinstance(clues, list):
        raise ValueError("parsed OCR output must contain a clues list")

    rooms = [[None for _ in range(size)] for _ in range(size)]
    objects = [[None for _ in range(size)] for _ in range(size)]

    for cell in cells:
        try:
            row = int(cell["row"]) - 1
            column = int(cell["col"]) - 1
            room = str(cell["room"]).strip().lower()
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid board cell: {cell!r}") from exc

        if not 0 <= row < size or not 0 <= column < size:
            raise ValueError(f"board cell is outside the {size}x{size} grid: {cell!r}")
        if rooms[row][column] is not None:
            raise ValueError(f"duplicate board cell at row {row + 1}, col {column + 1}")

        rooms[row][column] = room
        objects[row][column] = _normalise_object(cell.get("object"))

    missing = [
        (row + 1, column + 1)
        for row in range(size)
        for column in range(size)
        if rooms[row][column] is None
    ]
    if missing:
        raise ValueError(f"board output is missing cells: {missing}")

    puzzle = {
        "grid_size": size,
        "rooms": rooms,
        "objects": objects,
        "blocked_objects": sorted(_normalise_object(obj) for obj in blocked_objects),
        "people": list(people) if people is not None else _people_from_clues(clues, victim),
        "victim": victim,
        "clues": clues,
    }
    validate_puzzle(puzzle)
    return puzzle


def validate_puzzle(puzzle: dict[str, Any]) -> None:
    """Validate an adapted puzzle and raise ValueError for malformed vision data."""
    size = puzzle["grid_size"]
    rooms = puzzle["rooms"]
    objects = puzzle["objects"]
    people = puzzle["people"]
    victim = puzzle["victim"]

    if len(rooms) != size or len(objects) != size:
        raise ValueError("rooms and objects must have grid_size rows")
    if any(len(row) != size for row in rooms + objects):
        raise ValueError("rooms and objects must form a square grid")
    if len(people) != size or len(set(people)) != size:
        raise ValueError(
            f"expected {size} different people, but received {people!r}; "
            "pass people=[...] to puzzle_from_vision if OCR missed a name"
        )
    if victim not in people:
        raise ValueError(f"victim {victim!r} is not in the people list")

    room_names = {room for row in rooms for room in row}
    object_names = {obj for row in objects for obj in row if obj is not None}

    for clue in puzzle["clues"]:
        if not isinstance(clue, list) or len(clue) < 3:
            raise ValueError(f"invalid clue: {clue!r}")

        person, relation, argument, *extra = clue
        if person not in people:
            raise ValueError(f"unknown person in clue: {clue!r}")
        if relation not in SUPPORTED_RELATIONS:
            raise ValueError(f"unknown relation in clue: {clue!r}")

        if relation in {"in room", "not in room"} and argument not in room_names:
            raise ValueError(f"unknown room in clue: {clue!r}")
        if relation in {"on top", "not on top"} and argument not in object_names:
            raise ValueError(f"unknown object in clue: {clue!r}")
        if relation in {"row", "column"} and (
            not isinstance(argument, int) or not 0 <= argument < size
        ):
            raise ValueError(f"grid index outside 0..{size - 1}: {clue!r}")
        if relation in PERSON_RELATIONS and argument not in people:
            raise ValueError(f"unknown person in clue: {clue!r}")
        if relation in {"beside", "not beside"} and (
            argument not in people and argument not in object_names
        ):
            raise ValueError(f"unknown person or object in clue: {clue!r}")
        if relation in DIRECTIONS and extra and (
            not isinstance(extra[0], int) or extra[0] <= 0
        ):
            raise ValueError(f"direction distance must be a positive integer: {clue!r}")


def _neighbors(row: int, column: int, size: int):
    for row_delta, column_delta in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        neighbor = row + row_delta, column + column_delta
        if 0 <= neighbor[0] < size and 0 <= neighbor[1] < size:
            yield neighbor


def allowed_cells(puzzle: dict[str, Any]) -> dict[str, set[tuple[int, int]]]:
    """Apply all unary clues before constructing the CP-SAT model."""
    size = puzzle["grid_size"]
    rooms = puzzle["rooms"]
    objects = puzzle["objects"]
    all_cells = {(row, col) for row in range(size) for col in range(size)}
    blocked = set(puzzle["blocked_objects"])
    free_cells = {
        (row, col)
        for row, col in all_cells
        if objects[row][col] not in blocked
    }
    allowed = {person: set(free_cells) for person in puzzle["people"]}

    for person, relation, argument, *_ in puzzle["clues"]:
        if relation == "in room":
            allowed[person] &= {
                cell for cell in all_cells if rooms[cell[0]][cell[1]] == argument
            }
        elif relation == "not in room":
            allowed[person] -= {
                cell for cell in all_cells if rooms[cell[0]][cell[1]] == argument
            }
        elif relation == "on top":
            allowed[person] &= {
                cell for cell in all_cells if objects[cell[0]][cell[1]] == argument
            }
        elif relation == "not on top":
            allowed[person] -= {
                cell for cell in all_cells if objects[cell[0]][cell[1]] == argument
            }
        elif relation in {"beside", "not beside"} and argument not in puzzle["people"]:
            beside_object = {
                (row, col)
                for row, col in all_cells
                if any(
                    objects[other_row][other_col] == argument
                    and rooms[other_row][other_col] == rooms[row][col]
                    for other_row, other_col in _neighbors(row, col, size)
                )
            }
            if relation == "beside":
                allowed[person] &= beside_object
            else:
                allowed[person] -= beside_object
        elif relation == "row":
            allowed[person] &= {cell for cell in all_cells if cell[0] == argument}
        elif relation == "column":
            allowed[person] &= {cell for cell in all_cells if cell[1] == argument}

    return allowed


def build_model(puzzle: dict[str, Any]):
    """Create the CP-SAT model and return it together with its variables."""
    validate_puzzle(puzzle)
    size = puzzle["grid_size"]
    people = puzzle["people"]
    victim = puzzle["victim"]
    rooms = puzzle["rooms"]
    room_names = sorted({room for row in rooms for room in row})
    room_ids = {room: index for index, room in enumerate(room_names)}
    flat_rooms = [room_ids[rooms[row][col]] for row in range(size) for col in range(size)]
    permitted = allowed_cells(puzzle)

    model = cp_model.CpModel()
    row_vars = {
        person: model.new_int_var(0, size - 1, f"row_{person}") for person in people
    }
    column_vars = {
        person: model.new_int_var(0, size - 1, f"column_{person}") for person in people
    }
    room_vars = {
        person: model.new_int_var(0, len(room_names) - 1, f"room_{person}")
        for person in people
    }

    model.add_all_different(row_vars.values())
    model.add_all_different(column_vars.values())

    for person in people:
        model.add_allowed_assignments(
            [row_vars[person], column_vars[person]],
            sorted(permitted[person]),
        )
        flat_index = model.new_int_var(0, size * size - 1, f"index_{person}")
        model.add(flat_index == size * row_vars[person] + column_vars[person])
        model.add_element(flat_index, flat_rooms, room_vars[person])

    for person, relation, argument, *extra in puzzle["clues"]:
        if relation == "same room":
            model.add(room_vars[person] == room_vars[argument])
        elif relation == "different room":
            model.add(room_vars[person] != room_vars[argument])
        elif relation == "alone":
            for other_person in people:
                if other_person != person:
                    model.add(room_vars[other_person] != room_vars[person])
        elif relation == "alone with":
            model.add(room_vars[person] == room_vars[argument])
            for other_person in people:
                if other_person not in {person, argument}:
                    model.add(room_vars[other_person] != room_vars[person])
        elif relation in DIRECTIONS:
            axis_name, sign = DIRECTIONS[relation]
            axis = row_vars if axis_name == "row" else column_vars
            difference = sign * (axis[person] - axis[argument])
            if extra:
                model.add(difference == extra[0])
            else:
                model.add(difference >= 1)
        elif relation == "beside" and argument in people:
            model.add(room_vars[person] == room_vars[argument])
            row_difference = model.new_int_var(-size, size, f"row_diff_{person}_{argument}")
            column_difference = model.new_int_var(
                -size, size, f"column_diff_{person}_{argument}"
            )
            abs_row_difference = model.new_int_var(
                0, size, f"abs_row_diff_{person}_{argument}"
            )
            abs_column_difference = model.new_int_var(
                0, size, f"abs_column_diff_{person}_{argument}"
            )
            model.add(row_difference == row_vars[person] - row_vars[argument])
            model.add(column_difference == column_vars[person] - column_vars[argument])
            model.add_abs_equality(abs_row_difference, row_difference)
            model.add_abs_equality(abs_column_difference, column_difference)
            model.add(abs_row_difference + abs_column_difference == 1)
        elif relation == "not beside" and argument in people:
            # With one person per row and column, this is already guaranteed.
            pass

    shares_victim_room = {}
    for person in people:
        if person == victim:
            continue
        shares_room = model.new_bool_var(f"shares_victim_room_{person}")
        model.add(room_vars[person] == room_vars[victim]).only_enforce_if(shares_room)
        model.add(room_vars[person] != room_vars[victim]).only_enforce_if(shares_room.negated())
        shares_victim_room[person] = shares_room
    model.add_exactly_one(shares_victim_room.values())

    variables = {
        "row": row_vars,
        "column": column_vars,
        "room": room_vars,
        "shares_victim_room": shares_victim_room,
    }
    return model, variables


def solve_puzzle(puzzle: dict[str, Any]) -> dict[str, Any] | None:
    """Solve one puzzle, returning a JSON-compatible result or None."""
    model, variables = build_model(puzzle)
    solver = cp_model.CpSolver()
    status = solver.solve(model)
    if status not in {cp_model.OPTIMAL, cp_model.FEASIBLE}:
        return None

    positions = {}
    for person in puzzle["people"]:
        row = solver.value(variables["row"][person])
        column = solver.value(variables["column"][person])
        positions[person] = {
            "row": row,
            "column": column,
            "room": puzzle["rooms"][row][column],
            "object": puzzle["objects"][row][column],
        }

    murderer = next(
        person
        for person, shares_room in variables["shares_victim_room"].items()
        if solver.value(shares_room)
    )
    return {
        "positions": positions,
        "murderer": murderer,
        "status": solver.status_name(status),
        "stats": {
            "wall_time_seconds": solver.wall_time,
            "branches": solver.num_branches,
            "conflicts": solver.num_conflicts,
        },
    }


class _SolutionCollector(cp_model.CpSolverSolutionCallback):
    def __init__(self, row_vars, column_vars, limit: int | None):
        super().__init__()
        self._row_vars = row_vars
        self._column_vars = column_vars
        self._limit = limit
        self.solutions = []

    def on_solution_callback(self):
        self.solutions.append(
            {
                person: {
                    "row": self.value(self._row_vars[person]),
                    "column": self.value(self._column_vars[person]),
                }
                for person in self._row_vars
            }
        )
        if self._limit is not None and len(self.solutions) >= self._limit:
            self.stop_search()


def find_all_solutions(
    puzzle: dict[str, Any],
    *,
    limit: int | None = None,
) -> list[dict[str, dict[str, int]]]:
    """Enumerate solutions, optionally stopping after ``limit`` results."""
    if limit is not None and limit <= 0:
        raise ValueError("limit must be a positive integer or None")

    model, variables = build_model(puzzle)
    solver = cp_model.CpSolver()
    solver.parameters.enumerate_all_solutions = True
    collector = _SolutionCollector(variables["row"], variables["column"], limit)
    solver.solve(model, collector)
    return collector.solutions


def solve_vision_outputs(
    board_output: dict[str, Any],
    parsed_ocr_output: dict[str, Any],
    **adapter_options,
) -> dict[str, Any] | None:
    """Adapt and solve the two vision outputs in one call."""
    puzzle = puzzle_from_vision(board_output, parsed_ocr_output, **adapter_options)
    return solve_puzzle(puzzle)


def main() -> None:
    parser = argparse.ArgumentParser(description="Solve a Murdoku image with CP-SAT")
    project_dir = Path(__file__).resolve().parents[1]
    parser.add_argument("--image", type=Path, default=project_dir / "data" / "puzzle1.png")
    parser.add_argument(
        "--templates",
        type=Path,
        default=project_dir / "data" / "templates",
    )
    parser.add_argument("--victim", default=DEFAULT_VICTIM)
    args = parser.parse_args()

    from cv.board_reader import analyze_murdoku
    from cv.ocr import parse_clues, read_clues

    board_output = analyze_murdoku(args.image, args.templates)
    parsed_ocr_output = parse_clues(read_clues(str(args.image)))
    result = solve_vision_outputs(
        board_output,
        parsed_ocr_output,
        victim=args.victim,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

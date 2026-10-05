from importlib import import_module

__all__ = [
    "allowed_cells",
    "build_model",
    "find_all_solutions",
    "puzzle_from_vision",
    "solve_puzzle",
    "solve_vision_outputs",
    "validate_puzzle",
]


def __getattr__(name):
    if name in __all__:
        return getattr(import_module(".solver", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

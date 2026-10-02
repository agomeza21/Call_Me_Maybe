import json
from typing import Any
import os


def load_json_file(path: str) -> list[dict[Any, Any]]:
    try:
        with open(path) as f:
            data: list[dict[Any, Any]] = json.load(f)
    except FileNotFoundError as e:
        raise ValueError(f"File not found: {path}") from e
    except OSError as e:
        raise ValueError(f"Cannot read file {path}: {e}") from e
    except json.JSONDecodeError as e:
        raise ValueError(f"In file {path}: {e}") from e
    return data


def save_json_file(path: str, data: list[dict[str, Any]]) -> None:
    try:
        if os.path.dirname(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    except OSError as e:
        raise ValueError(f"In file {path}: {e}") from e

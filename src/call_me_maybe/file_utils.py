"""Reading and writing of the JSON files used by the program."""

import json
from typing import Any
import os


def load_json_file(path: str) -> list[dict[Any, Any]]:
    """Reads and parses a JSON file.

    Only the JSON syntax is checked here. The shape of the content is
    validated later by parse_functions and parse_tests.

    Args:
        path: Path to the JSON file.

    Returns:
        The parsed JSON content.

    Raises:
        ValueError: If the file does not exist, cannot be read or is
            not valid JSON. The message includes the path.
    """
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
    """Writes data as indented JSON, creating missing directories.

    Args:
        path: Path of the output file.
        data: Content to write.

    Raises:
        ValueError: If the directory or the file cannot be created or
            written. The message includes the path.
    """

    try:
        if os.path.dirname(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    except OSError as e:
        raise ValueError(f"In file {path}: {e}") from e

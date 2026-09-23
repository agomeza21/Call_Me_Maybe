import json
import os


def load_json_file(path: str) -> list[dict]:
    try:
        with open(path) as f:
            data = json.load(f)
    except FileNotFoundError as e:
        raise ValueError(f"File not found: {path}") from e
    except json.JSONDecodeError as e:
        raise ValueError(f"In file {path}: {e}") from e
    return data


def save_json_file(path: str, data: list[dict]) -> None:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    except OSError as e:
        raise ValueError(f"In file {path}: {e}") from e

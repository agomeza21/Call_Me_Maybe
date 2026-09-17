import json


def load_json_file(path: str) -> list[dict]:
    try:
        with open(path) as f:
            data = json.load(f)
    except FileNotFoundError as e:
        raise ValueError(f"File not found: {path}") from e
    except json.JSONDecodeError as e:
        raise ValueError(f"In file {path}: {e}") from e
    return data

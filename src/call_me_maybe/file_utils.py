import json


def load_json_file(path: str) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    return data

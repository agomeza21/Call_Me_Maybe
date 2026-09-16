import json


def load_vocab(vocab_path: str) -> dict:
    with open(vocab_path) as f:
        vocab = json.load(f)
    return vocab


def get_valid_function_names(functions: list[dict]) -> list[str]:
    functions_name = []
    for func in functions:
        functions_name.append(func["name"])
    return functions_name

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


def invert_vocab(vocab: dict) -> dict:
    id_to_token = {}
    for token, id in vocab.items():
        id_to_token[id] = token
    return id_to_token


def is_valid_prefix(candidate: str, valid_names: list[str]) -> bool:
    for valid in valid_names:
        if valid.startswith(candidate) is True:
            return True
        else:
            continue
    return False

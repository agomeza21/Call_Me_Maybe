import json
from .models import FunctionDefinition


def load_vocab(vocab_path: str) -> dict:
    with open(vocab_path) as f:
        vocab = json.load(f)
    return vocab


def get_valid_function_names(functions: list[FunctionDefinition]) -> list[str]:
    functions_name = []
    for func in functions:
        functions_name.append(func.name)
    return functions_name


def invert_vocab(vocab: dict) -> dict:
    id_to_token = {}
    for token, id in vocab.items():
        id_to_token[id] = token
    return id_to_token


def is_valid_prefix(candidate: str, valid_names: list[str]) -> bool:
    for valid in valid_names:
        if valid.startswith(candidate):
            return True
        else:
            continue
    return False


def mask_logits(logits: list[float], id_to_token: dict,
                generated_text: str, valid_names: list[str]) -> list[float]:
    modified_logits = []
    for id_token, logit in enumerate(logits):
        if id_token not in id_to_token:
            modified_logits.append(float("-inf"))
            continue
        candidate = generated_text + id_to_token[id_token]
        if is_valid_prefix(candidate, valid_names):
            modified_logits.append(logit)
        else:
            modified_logits.append(float("-inf"))
    return modified_logits


def count_matching_prefixes(text: str, valid_names: list[str]) -> int:
    counter = 0
    for name in valid_names:
        if name.startswith(text):
            counter += 1
    return counter


def is_valid_number_char(candidate: str, allow_decimal: bool) -> bool:
    if allow_decimal:
        valid_chars = {
            "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ".", "-"}
    else:
        valid_chars = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "-"}
    for char in candidate:
        if char not in valid_chars:
            return False
    return True


def mask_logits_number(logits: list[float], id_to_token: dict,
                       generated_text: str,
                       allow_decimal: bool) -> list[float]:
    modified_logits = []
    for id_token, logit in enumerate(logits):
        if id_token not in id_to_token:
            modified_logits.append(float("-inf"))
            continue
        candidate = generated_text + id_to_token[id_token]
        if is_valid_number_char(candidate, allow_decimal):
            modified_logits.append(logit)
        else:
            modified_logits.append(float("-inf"))
    return modified_logits

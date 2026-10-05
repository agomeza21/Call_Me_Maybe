import json
import numpy
from typing import Callable
from .models import FunctionDefinition


def load_vocab(vocab_path: str) -> dict[str, int]:
    try:
        with open(vocab_path) as f:
            vocab: dict[str, int] = json.load(f)
    except FileNotFoundError as e:
        raise ValueError(f"File not found: {vocab_path}") from e
    except OSError as e:
        raise ValueError(f"Cannot read file {vocab_path}: {e}") from e
    except json.JSONDecodeError as e:
        raise ValueError(f"In file {vocab_path}: {e}") from e
    return vocab


def get_valid_function_names(functions: list[FunctionDefinition]) -> list[str]:
    functions_name = []
    for func in functions:
        functions_name.append(func.name)
    return functions_name


def invert_vocab(vocab: dict[str, int]) -> dict[int, str]:
    id_to_token = {}
    for token_str, token_id in vocab.items():
        id_to_token[token_id] = token_str
    return id_to_token


def is_valid_prefix(candidate: str, valid_names: list[str]) -> bool:
    for valid in valid_names:
        if valid.startswith(candidate):
            return True
        else:
            continue
    return False


def count_matching_prefixes(text: str, valid_names: list[str]) -> int:
    counter = 0
    for name in valid_names:
        if name.startswith(text):
            counter += 1
    return counter


def is_valid_number_char(candidate: str, allow_decimal: bool) -> bool:
    if candidate.startswith('Ġ') or candidate.startswith(' '):
        candidate = candidate[1:]
    if not candidate:
        return True
    has_decimal_point = False
    for i, char in enumerate(candidate):
        if char == '-':
            if i != 0:
                return False
        elif char == '.':
            if not allow_decimal:
                return False
            if has_decimal_point:
                return False
            has_decimal_point = True
        elif not char.isdigit():
            return False
    return True


def is_valid_string(candidate: str) -> int:
    # Estados:
    # 0 = Inválido
    # 1 = Válido (texto normal)
    # 2 = Comilla o carácter escapado correctamente (\", \n, etc.)
    # 3 = Pendiente de escape (termina en '\')
    # 4 = Comilla de cierre de JSON alcanzada

    valid_escaped_chars = ['"', '\\', '/', 'b', 'f', 'n', 'r', 't', 'u']
    escaped = False
    has_escaped_quote = False
    for char in candidate:
        if escaped:
            if char not in valid_escaped_chars:
                return 0
            if char == '"':
                has_escaped_quote = True
            escaped = False
        else:
            if char == '\\':
                escaped = True
            elif char == '"':
                return 4
    if escaped:
        return 3
    if has_escaped_quote:
        return 2
    return 1


def select_valid_token(masked_logits: list[float]) -> int:
    best_id = int(numpy.argmax(masked_logits))
    if masked_logits[best_id] == float("-inf"):
        raise ValueError(
            "constrained decoding found no valid token to continue "
            "generation")
    return best_id


def mask_logits(logits: list[float], id_to_token: dict[int, str],
                generated_text: str,
                is_valid: Callable[[str], int]) -> list[float]:
    modified_logits = []
    for candidate_id, logit in enumerate(logits):
        if candidate_id not in id_to_token:
            modified_logits.append(float("-inf"))
            continue
        candidate = generated_text + id_to_token[candidate_id]
        valid_result = is_valid(candidate)
        if valid_result > 0:
            modified_logits.append(logit)
        else:
            modified_logits.append(float("-inf"))
    return modified_logits

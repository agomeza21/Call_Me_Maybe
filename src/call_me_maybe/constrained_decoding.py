"""Helpers for constrained decoding.

At every generation step the model returns one logit per token of its
vocabulary. The functions here decide which tokens are allowed at
that moment (prefix of a valid name, valid number characters, valid
JSON string content...). Disallowed tokens get -inf, so the argmax can
only ever pick an allowed one.
"""

import json
import numpy
from typing import Callable
from .models import FunctionDefinition


def load_vocab(vocab_path: str) -> dict[str, int]:
    """Loads the tokenizer vocabulary from a JSON file.

    The file maps each token string to its integer id. Token strings
    are in byte-level BPE form, not plain text: a leading space is
    written as 'Ġ'.

    Args:
        vocab_path: Path to the vocabulary JSON file.

    Returns:
        Dict mapping token string to token id.

    Raises:
        ValueError: If the file is missing, cannot be read or does not
            contain valid JSON.
    """

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
    """Returns the names of all the available functions.

    Args:
        functions: Parsed function definitions.

    Returns:
        List of names, in the same order as the definitions.
    """

    functions_name = []
    for func in functions:
        functions_name.append(func.name)
    return functions_name


def invert_vocab(vocab: dict[str, int]) -> dict[int, str]:
    """Reverses the vocabulary: token id to token string.

    The model works with ids but the validators work with text, so
    this lets us turn any candidate id into its text.

    Args:
        vocab: Dict mapping token string to token id.

    Returns:
        Dict mapping token id to token string.
    """

    id_to_token = {}
    for token_str, token_id in vocab.items():
        id_to_token[token_id] = token_str
    return id_to_token


def is_valid_prefix(candidate: str, valid_names: list[str]) -> bool:
    """Checks that candidate can still grow into one of the valid names.

    Used for function names and for the boolean literals: a token is
    allowed only if the text stays a prefix of at least one valid
    name.

    Args:
        candidate: Text generated so far plus the token being tested.
        valid_names: Strings the final text may be equal to.

    Returns:
        True if some valid name starts with candidate.
    """

    for valid in valid_names:
        if valid.startswith(candidate):
            return True
        else:
            continue
    return False


def count_matching_prefixes(text: str, valid_names: list[str]) -> int:
    """Counts how many valid names start with text.

    Combined with an exact match it tells us when a value is complete
    without ambiguity: the text equals a name and no other name starts
    with it.
    A name that is a prefix of another one (for example fn_add and
    fn_add_numbers) matches two names. _generate_function_name
    handles that case by letting the model choose between
    continuing the name and closing the quote.

    Args:
        text: Text generated so far.
        valid_names: Strings the final text may be equal to.

    Returns:
        Number of valid names that start with text.
    """

    counter = 0
    for name in valid_names:
        if name.startswith(text):
            counter += 1
    return counter


def is_valid_number_char(candidate: str, allow_decimal: bool) -> bool:
    """Checks that candidate is a valid start of a number.

    One leading 'Ġ' or space is ignored: the model may emit the space
    before the number inside its token (for example 'Ġ-'). The rest
    may only contain digits, a '-' as the very first character and,
    if allow_decimal is True, a single '.'. The empty string is valid
    (nothing generated yet).

    Args:
        candidate: Text generated so far plus the token being tested.
        allow_decimal: True for "number" parameters, False for
            "integer" parameters.

    Returns:
        True if candidate can still be completed into a valid number.
    """

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
    """Classifies the content of a JSON string being generated.

    The candidate is the text after the opening quote. Backslash
    escapes are checked, and the first unescaped double quote is
    treated as the closing quote of the JSON string.

    Args:
        candidate: Text generated so far plus the token being tested.

    Returns:
        0: invalid (a backslash followed by a character that cannot
            be escaped).
        1: valid, plain text.
        2: valid, and it contains an escaped double quote.
        3: valid, but it ends with a lone backslash (the escape is
            still pending).
        4: an unescaped double quote was found, so the string ends
            there. Whatever follows that quote inside the same token
            is discarded later by _text_before_closing_quote.
    """

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
    """Picks the allowed token with the highest logit.

    Args:
        masked_logits: Logits where disallowed tokens are -inf.

    Returns:
        Id of the token with the highest logit.

    Raises:
        ValueError: If every token is -inf (no valid token exists).
    """

    best_id = int(numpy.argmax(masked_logits))
    if masked_logits[best_id] == float("-inf"):
        raise ValueError(
            "constrained decoding found no valid token to continue "
            "generation")
    return best_id


def mask_logits(logits: list[float], id_to_token: dict[int, str],
                generated_text: str,
                is_valid: Callable[[str], int]) -> list[float]:
    """Sets to -inf the logit of every token that is not allowed.

    For each token id, its text is appended to generated_text and
    passed to is_valid. A result greater than 0 keeps the original
    logit and 0 masks it. Ids missing from id_to_token are always
    masked.

    Args:
        logits: Raw logits from the model, one per token id.
        id_to_token: Mapping token id to token string.
        generated_text: Text generated so far for the current value.
        is_valid: Validator that receives the candidate text and
            returns 0 if it is invalid and a value above 0 otherwise.

    Returns:
        New list of the same length as logits where only the allowed
        tokens keep their logit.
    """

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

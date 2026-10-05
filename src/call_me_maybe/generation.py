import numpy
import json
import sys
from typing import Any
from llm_sdk import Small_LLM_Model
from functools import partial
from .models import TestPrompt, FunctionDefinition, FunctionCallResult
from .constrained_decoding import (invert_vocab, get_valid_function_names,
                                   count_matching_prefixes,
                                   is_valid_number_char,
                                   is_valid_string, is_valid_prefix,
                                   mask_logits, select_valid_token)


def build_prompt(functions: list[FunctionDefinition], prompt: str) -> str:
    functions_dict = []
    for f in functions:
        functions_dict.append(f.model_dump())
    functions_text = json.dumps(functions_dict, indent=2)
    instructions = ("Given the following functions, respond with the name of "
                    "the function to call and its parameters, in JSON format. "
                    "Choose the function whose description best matches the "
                    "user's overall intent, not just individual words in the "
                    "prompt. Prioritize \\\\ over other escape chars.")
    result = (
        f"<|im_start|>system\n{instructions}\n\n{functions_text}<|im_end|>\n"
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    return result


def build_error_result(prompt: str, warning: str) -> dict[str, Any]:
    print(f"Warning: {warning}", file=sys.stderr)
    result = FunctionCallResult(prompt=prompt, name="ERROR", parameters={})
    return result.model_dump()


def _append_tokens(model: Small_LLM_Model, token_ids: list[int],
                   text_to_encode: str) -> None:
    encoded_token_ids = model.encode(text_to_encode)[0].tolist()
    token_ids.extend(encoded_token_ids)


def _generate_function_name(model: Small_LLM_Model, token_ids: list[int],
                            id_to_token: dict[int, str],
                            valid_names: list[str]) -> str:
    max_func_name_tokens = 200
    prompt_length = len(token_ids)
    for _ in range(max_func_name_tokens):
        generated_text = model.decode(token_ids[prompt_length:])
        if (generated_text in valid_names and
                count_matching_prefixes(generated_text,
                                        valid_names) == 1):
            return str(generated_text)

        logits = model.get_logits_from_input_ids(token_ids)
        is_valid = partial(is_valid_prefix, valid_names=valid_names)
        masked_logits = mask_logits(logits, id_to_token,
                                    generated_text, is_valid)
        selected_token_id = select_valid_token(masked_logits)
        token_ids.append(int(selected_token_id))

    return ""


def _text_before_closing_quote(text: str) -> str:
    escaped = False
    for i, char in enumerate(text):
        if escaped:
            escaped = False
        elif char == '\\':
            escaped = True
        elif char == '"':
            return text[:i]
    return text


def _generate_string_param(model: Small_LLM_Model, token_ids: list[int],
                           id_to_token: dict[int, str]) -> str | None:
    max_string_tokens = 200
    _append_tokens(model, token_ids, '"')
    value_text = ""
    value_start = len(token_ids)

    for _ in range(max_string_tokens):
        logits = model.get_logits_from_input_ids(token_ids)
        masked_logits = mask_logits(logits, id_to_token,
                                    value_text, is_valid_string)
        id_token = int(select_valid_token(masked_logits))
        raw_token = id_to_token.get(id_token, "")
        candidate = value_text + raw_token
        state = is_valid_string(candidate)

        if state == 4:
            token_text = model.decode([id_token])
            content_before = _text_before_closing_quote(token_text)
            if content_before:
                _append_tokens(model, token_ids, content_before)
                value_text = model.decode(token_ids[value_start:])
            break

        token_ids.append(id_token)
        value_text = model.decode(token_ids[value_start:])

    _append_tokens(model, token_ids, '"')

    try:
        decoded_val = json.loads(f'"{value_text}"')
        return str(decoded_val)
    except json.JSONDecodeError:
        return None


def _generate_boolean_param(model: Small_LLM_Model, token_ids: list[int],
                            id_to_token: dict[int, str]) -> bool | None:
    max_boolean_tokens = 20
    boolean_values = ["true", "false"]
    value_text = ""
    value_start = len(token_ids)
    for _ in range(max_boolean_tokens):
        value_text = model.decode(token_ids[value_start:])
        if (value_text in boolean_values and
                count_matching_prefixes(value_text,
                                        boolean_values) == 1):
            break

        logits = model.get_logits_from_input_ids(token_ids)
        is_valid = partial(is_valid_prefix, valid_names=boolean_values)
        masked_logits = mask_logits(logits, id_to_token, value_text, is_valid)
        id_token = select_valid_token(masked_logits)
        token_ids.append(int(id_token))

    if value_text not in boolean_values:
        return None

    return value_text == "true"


def _generate_number_param(model: Small_LLM_Model, token_ids: list[int],
                           id_to_token: dict[int, str],
                           param_type_name: str) -> int | float | None:
    max_number_tokens = 20
    value_text = ""
    value_start = len(token_ids)
    allow_decimal = param_type_name == "number"
    for _ in range(max_number_tokens):
        logits = model.get_logits_from_input_ids(token_ids)
        id_token_raw = numpy.argmax(logits)
        if int(id_token_raw) not in id_to_token:
            break

        raw_token = id_to_token[int(id_token_raw)]
        raw_candidate = value_text + raw_token

        if is_valid_number_char(raw_candidate, allow_decimal):
            is_valid = partial(is_valid_number_char,
                               allow_decimal=allow_decimal)
            masked_logits = mask_logits(logits, id_to_token,
                                        value_text, is_valid)
            id_token = select_valid_token(masked_logits)
            token_ids.append(int(id_token))
            value_text = model.decode(token_ids[value_start:])
        else:
            break

    if not value_text:
        return None

    try:
        if allow_decimal:
            return float(value_text)
        else:
            return int(value_text)
    except ValueError:
        return None


def _process_single_test(test: TestPrompt, functions: list[FunctionDefinition],
                         valid_names: list[str], id_to_token: dict[int, str],
                         model: Small_LLM_Model) -> dict[str, Any]:
    if not test.prompt.strip():
        return build_error_result(test.prompt, "prompt is empty or blank")

    prompt = build_prompt(functions, test.prompt)
    ids = model.encode(prompt)
    token_ids: list[int] = ids[0].tolist()

    _append_tokens(model, token_ids, '{"name": "')

    function_name = _generate_function_name(
        model, token_ids, id_to_token, valid_names)

    selected_function = None
    for func in functions:
        if func.name == function_name:
            selected_function = func
            break

    if selected_function is None:
        return build_error_result(test.prompt, f"could not determine a valid "
                                  f"function for prompt: {test.prompt}")

    _append_tokens(model, token_ids, '", "parameters": {')

    parameters_dict: dict[str, str | int | float | bool] = {}

    for index, (param_name, param_type) in (
            enumerate(selected_function.parameters.items())):
        if index > 0:
            _append_tokens(model, token_ids, ", ")
        _append_tokens(model, token_ids, f'"{param_name}":')

        parsed_value: str | int | float | bool | None = None
        if param_type.type == "string":
            parsed_value = _generate_string_param(model, token_ids,
                                                  id_to_token)
        elif param_type.type == "boolean":
            parsed_value = _generate_boolean_param(model, token_ids,
                                                   id_to_token)
        else:
            parsed_value = _generate_number_param(model, token_ids,
                                                  id_to_token, param_type.type)

        if parsed_value is None:
            return build_error_result(test.prompt, f"could not generate a "
                                      f"valid value for parameter "
                                      f"'{param_name}' in prompt: "
                                      f"{test.prompt}")

        parameters_dict[param_name] = parsed_value

    _append_tokens(model, token_ids, "}")

    result = FunctionCallResult(
        prompt=test.prompt,
        name=function_name,
        parameters=parameters_dict
    )
    return result.model_dump()


def generate_function_calls(vocab: dict[str, int],
                            functions: list[FunctionDefinition],
                            tests: list[TestPrompt],
                            model: Small_LLM_Model) -> list[dict[str, Any]]:
    results = []
    id_to_token = invert_vocab(vocab)
    valid_names = get_valid_function_names(functions)

    for test in tests:
        try:
            test_result = _process_single_test(test, functions, valid_names,
                                               id_to_token, model)
            results.append(test_result)
        except Exception as e:
            results.append(build_error_result(
                test.prompt, f"unexpected error while processing "
                f"prompt '{test.prompt}': {e}"))
    return results

import numpy
import json
from typing import Any
from llm_sdk import Small_LLM_Model
from functools import partial
from .models import TestPrompt, FunctionDefinition, FunctionCallResult
from .constrained_decoding import (invert_vocab, get_valid_function_names,
                                   count_matching_prefixes,
                                   is_valid_number_char,
                                   is_valid_string, is_valid_prefix,
                                   mask_logits, is_balanced)


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
    print(f"Warning: {warning}")
    result = FunctionCallResult(prompt=prompt, name="ERROR", parameters={})
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
            prompt = build_prompt(functions, test.prompt)
            ids = model.encode(prompt)
            int_list: list[int] = ids[0].tolist()
            forced_names_ids = model.encode('{"name": "')[0].tolist()
            int_list.extend(forced_names_ids)
            prompt_length = len(int_list)
            function_name = ""
            for _ in range(50):
                generated_text = model.decode(int_list[prompt_length:])
                if (generated_text in valid_names and
                        count_matching_prefixes(generated_text,
                                                valid_names) == 1):
                    function_name = generated_text
                    forced_params_ids = (model.encode
                                         ('", "parameters": {')[0].tolist())
                    int_list.extend(forced_params_ids)
                    break
                logits = model.get_logits_from_input_ids(int_list)
                is_valid = partial(is_valid_prefix, valid_names=valid_names)
                masked_logits = mask_logits(logits, id_to_token,
                                            generated_text, is_valid)
                id_token = numpy.argmax(masked_logits)
                int_list.append(int(id_token))
            selected_function = None
            for func in functions:
                if func.name == function_name:
                    selected_function = func
                    break
            if selected_function is None:
                results.append(build_error_result(test.prompt, f"could not "
                                                  f"determine a valid function"
                                                  f" for prompt"
                                                  f": {test.prompt}"))
                continue
            parameters_dict: dict[str, str | int | float | bool] = {}
            generation_failed = False
            for index, (param_name, param_type) in (
                    enumerate(selected_function.parameters.items())):
                if index > 0:
                    forced_comma_ids = model.encode(", ")[0].tolist()
                    int_list.extend(forced_comma_ids)
                forced_param_name_ids = (model.
                                         encode(f'"{param_name}": ')[0].
                                         tolist())
                int_list.extend(forced_param_name_ids)
                if param_type.type == "string":
                    forced_quotes_ids = model.encode('"')[0].tolist()
                    int_list.extend(forced_quotes_ids)
                    value_text = ""
                    value_start = len(int_list)
                    for _ in range(20):
                        logits = model.get_logits_from_input_ids(int_list)
                        id_token_raw = numpy.argmax(logits)
                        if int(id_token_raw) not in id_to_token:
                            break
                        raw_token = id_to_token[int(id_token_raw)]
                        raw_candidate = value_text + raw_token

                        is_closing_quote = False
                        if '"' in raw_token:
                            if not raw_token.endswith('\\"'):
                                if not is_valid_string(raw_candidate):
                                    is_closing_quote = True
                        should_stop = False
                        if is_closing_quote:
                            if is_balanced(value_text):
                                should_stop = True
                        if should_stop:
                            content_before = raw_token.split('"')[0]
                            if content_before:
                                leftover_ids = (model.
                                                encode(content_before)[0].
                                                tolist())
                                int_list.extend(leftover_ids)
                                value_text = value_text + content_before
                            break
                        masked_logits = mask_logits(logits, id_to_token,
                                                    value_text,
                                                    is_valid_string)
                        id_token = numpy.argmax(masked_logits)
                        int_list.append(int(id_token))
                        value_text = model.decode(int_list[value_start:])
                    forced_quotes_ids = model.encode('"')[0].tolist()
                    int_list.extend(forced_quotes_ids)
                    try:
                        parameters_dict[param_name] = json.loads(
                            f'"{value_text}"')
                    except json.JSONDecodeError:
                        generation_failed = True
                        break
                elif param_type.type == "boolean":
                    boolean_values = ["true", "false"]
                    value_text = ""
                    value_start = len(int_list)
                    for _ in range(20):
                        value_text = model.decode(int_list[value_start:])
                        if (value_text in boolean_values and
                                count_matching_prefixes(value_text,
                                                        boolean_values) == 1):
                            break
                        logits = model.get_logits_from_input_ids(int_list)
                        is_valid = partial(is_valid_prefix,
                                           valid_names=boolean_values)
                        masked_logits = mask_logits(logits, id_to_token,
                                                    value_text, is_valid)
                        id_token = numpy.argmax(masked_logits)
                        int_list.append(int(id_token))
                    if value_text not in boolean_values:
                        generation_failed = True
                        break
                    parameters_dict[param_name] = value_text == "true"
                else:
                    value_text = ""
                    value_start = len(int_list)
                    allow_decimal = param_type.type == "number"
                    for _ in range(10):
                        logits = model.get_logits_from_input_ids(int_list)
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
                            id_token = numpy.argmax(masked_logits)
                            int_list.append(int(id_token))
                            value_text = model.decode(int_list[value_start:])
                        else:
                            break
                    if not value_text:
                        generation_failed = True
                        break
                    try:
                        if allow_decimal:
                            parameters_dict[param_name] = float(value_text)
                        else:
                            parameters_dict[param_name] = int(value_text)
                    except ValueError:
                        generation_failed = True
                        break
            if generation_failed:
                results.append(build_error_result(test.prompt, f"could not "
                                                  f"generate a valid value for"
                                                  f" parameter '{param_name}' "
                                                  f"in prompt: {test.prompt}"))
                continue
            forced_closing_ids = model.encode("}")[0].tolist()
            int_list.extend(forced_closing_ids)
            result = FunctionCallResult(
                prompt=test.prompt,
                name=function_name,
                parameters=parameters_dict
            )
            results.append(result.model_dump())
        except Exception as e:
            results.append(build_error_result(
                test.prompt, f"unexpected error while processing "
                f"prompt '{test.prompt}': {e}"))
    return results

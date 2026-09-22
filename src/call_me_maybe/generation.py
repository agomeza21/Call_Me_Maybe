import numpy
import json
from llm_sdk import Small_LLM_Model
from .models import TestPrompt, FunctionDefinition
from .constrained_decoding import (invert_vocab,
                                   get_valid_function_names, mask_logits_names,
                                   count_matching_prefixes,
                                   mask_logits_number, is_valid_number_char,
                                   is_valid_string, mask_logits_string)


def build_prompt(functions: list[FunctionDefinition], prompt: str) -> str:
    functions_dict = []
    for f in functions:
        functions_dict.append(f.model_dump())
    functions_text = json.dumps(functions_dict, indent=2)
    instructions = ("Given the following functions, respond with the name of "
                    "the function to call and its parameters, in JSON format."
                    "Choose the function whose description best matches the "
                    "user's overall intent, not just individual words in the "
                    "prompt")
    result = f"{instructions}\n\n{functions_text}\n\n{prompt}"
    return result


def generate_function_calls(vocab: dict, functions: list[FunctionDefinition],
                            tests: list[TestPrompt],
                            model: Small_LLM_Model) -> list[dict]:
    results = []
    id_to_token = invert_vocab(vocab)
    valid_names = get_valid_function_names(functions)
    for test in tests:
        prompt = build_prompt(functions, test.prompt)
        ids = model.encode(prompt)
        int_list: list = ids[0].tolist()
        forced_names_ids = model.encode('{"name": "')[0].tolist()
        int_list.extend(forced_names_ids)
        prompt_length = len(int_list)
        function_name = ""
        for _ in range(50):
            generated_text = model.decode(int_list[prompt_length:])
            if (generated_text in valid_names and
                    count_matching_prefixes(generated_text, valid_names) == 1):
                function_name = generated_text
                forced_params_ids = (model.encode
                                     ('", "parameters": {')[0].tolist())
                int_list.extend(forced_params_ids)
                break
            logits = model.get_logits_from_input_ids(int_list)
            masked_logits = mask_logits_names(logits, id_to_token,
                                              generated_text, valid_names)
            id_token = numpy.argmax(masked_logits)
            int_list.append(int(id_token))
        selected_function = None
        for func in functions:
            if func.name == function_name:
                selected_function = func
                break
        parameters_dict = {}
        for index, (param_name, param_type) in (
                enumerate(selected_function.parameters.items())):
            if index > 0:
                forced_comma_ids = model.encode(", ")[0].tolist()
                int_list.extend(forced_comma_ids)
            forced_param_name_ids = (model.
                                     encode(f'"{param_name}": ')[0].tolist())
            int_list.extend(forced_param_name_ids)
            if param_type.type == "string":
                forced_quotes_ids = model.encode('"')[0].tolist()
                int_list.extend(forced_quotes_ids)
                value_text = ""
                value_start = len(int_list)
                for _ in range(20):
                    logits = model.get_logits_from_input_ids(int_list)
                    id_token_raw = numpy.argmax(logits)
                    raw_token = id_to_token[int(id_token_raw)]
                    raw_candidate = value_text + raw_token
                    if is_valid_string(raw_candidate):
                        masked_logits = mask_logits_string(logits, id_to_token,
                                                           value_text)
                        id_token = numpy.argmax(masked_logits)
                        int_list.append(int(id_token))
                        value_text = model.decode(int_list[value_start:])
                    else:
                        break
                forced_quotes_ids = model.encode('"')[0].tolist()
                int_list.extend(forced_quotes_ids)
                parameters_dict[param_name] = value_text
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
                    masked_logits = mask_logits_names(logits, id_to_token,
                                                      value_text,
                                                      boolean_values)
                    id_token = numpy.argmax(masked_logits)
                    int_list.append(int(id_token))
                parameters_dict[param_name] = value_text == "true"
            else:
                value_text = ""
                value_start = len(int_list)
                allow_decimal = param_type.type == "number"
                for _ in range(10):
                    logits = model.get_logits_from_input_ids(int_list)
                    id_token_raw = numpy.argmax(logits)
                    raw_token = id_to_token[int(id_token_raw)]
                    raw_candidate = value_text + raw_token
                    if is_valid_number_char(raw_candidate, allow_decimal):
                        masked_logits = mask_logits_number(logits, id_to_token,
                                                           value_text,
                                                           allow_decimal)
                        id_token = numpy.argmax(masked_logits)
                        int_list.append(int(id_token))
                        value_text = model.decode(int_list[value_start:])
                    else:
                        break
                if allow_decimal:
                    parameters_dict[param_name] = float(value_text)
                else:
                    parameters_dict[param_name] = int(value_text)
        forced_closing_ids = model.encode("}")[0].tolist()
        int_list.extend(forced_closing_ids)
        res_dict = {
            "prompt": test.prompt,
            "name": function_name,
            "parameters": parameters_dict
        }
        results.append(res_dict)
    return results

import numpy
import json
from llm_sdk import Small_LLM_Model
from .models import TestPrompt, FunctionDefinition
from .constrained_decoding import (invert_vocab,
                                   get_valid_function_names, mask_logits,
                                   count_matching_prefixes)


def build_prompt(functions: list[FunctionDefinition], prompt: str) -> str:
    functions_dict = []
    for f in functions:
        functions_dict.append(f.model_dump())
    functions_text = json.dumps(functions_dict, indent=2)
    instructions = ("Given the following functions, respond with the name of "
                    "the function to call and its parameters, in JSON format")
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
                forced_params_ids = model.encode('", "parameters": {')[0].tolist()
                int_list.extend(forced_params_ids)
                print(model.decode(int_list))
                break
            logits = model.get_logits_from_input_ids(int_list)
            masked_logits = mask_logits(logits, id_to_token,
                                        generated_text, valid_names)
            id_token = numpy.argmax(masked_logits)
            int_list.append(int(id_token))

        res_dict = {
            "prompt": test.prompt,
            "name": function_name,
            "parameters": {}
        }
        results.append(res_dict)
    return results

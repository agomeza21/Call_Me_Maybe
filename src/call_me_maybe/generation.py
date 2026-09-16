import numpy
import json
from llm_sdk import Small_LLM_Model


def build_prompt(functions: list[dict], prompt: str) -> str:
    functions_text = json.dumps(functions, indent=2)
    instructions = ("Given the following functions, respond with the name of"
                    "the function to call and its parameters, in JSON format")
    result = f"{instructions}\n\n{functions_text}\n\n{prompt}"
    return result


def generate_function_calls(functions: list[dict], tests: list[dict],
                            model: Small_LLM_Model) -> list[dict]:
    results = []
    for test in tests:
        prompt = build_prompt(functions, test["prompt"])
        ids = model.encode(prompt)

        int_list: list = ids[0].tolist()

        fixed_start = '{"name": "'
        fixed_ids = model.encode(fixed_start)
        fixed_list = fixed_ids[0].tolist()
        int_list = int_list + fixed_list

        name_text = ""
        for _ in range(50):
            logits = model.get_logits_from_input_ids(int_list)
            id_token = numpy.argmax(logits)
            int_list.append(int(id_token))

        res = model.decode(int_list)

        res_dict = {"prompt": test["prompt"], "result": res}
        results.append(res_dict)
    return results

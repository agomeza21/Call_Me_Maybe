import argparse
import sys
from llm_sdk import Small_LLM_Model
from call_me_maybe.file_utils import load_json_file
from call_me_maybe.generation import generate_function_calls

parser = argparse.ArgumentParser()

parser.add_argument("--functions_definition",
                    default="data/input/functions_definition.json")
parser.add_argument("--input",
                    default="data/input/function_calling_tests.json")
parser.add_argument("--output",
                    default="data/output/function_calling_results.json")

args = parser.parse_args()

model = Small_LLM_Model()

try:
    functions = load_json_file(args.functions_definition)
    tests = load_json_file(args.input)
except (FileNotFoundError, ValueError) as e:
    print(e)
    sys.exit(1)

result = generate_function_calls(functions, tests, model)
print(result)

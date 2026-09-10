import argparse
import json
import sys

parser = argparse.ArgumentParser()

parser.add_argument("--functions_definition",
                    default="data/input/functions_definition.json")
parser.add_argument("--input",
                    default="data/input/function_calling_tests.json")
parser.add_argument("--output",
                    default="data/output/function_calling_results.json")

args = parser.parse_args()


def load_json_file(path: str) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    return data


try:
    functions = load_json_file(args.functions_definition)
    tests = load_json_file(args.input)
except (FileNotFoundError, ValueError) as e:
    print(e)
    sys.exit(1)

print(functions)
print()
print(tests)

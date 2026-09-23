import argparse
import sys
from llm_sdk import Small_LLM_Model
from call_me_maybe.file_utils import load_json_file, save_json_file
from call_me_maybe.generation import generate_function_calls
from call_me_maybe.constrained_decoding import load_vocab
from call_me_maybe.models import parse_functions, parse_tests

parser = argparse.ArgumentParser()

parser.add_argument("--functions_definition",
                    default="data/input/functions_definition.json")
parser.add_argument("--input",
                    default="data/input/function_calling_tests.json")
parser.add_argument("--output",
                    default="data/output/function_calling_results.json")

args = parser.parse_args()

model = Small_LLM_Model()
vocab_path = model.get_path_to_vocab_file()
result = []
try:
    vocab = load_vocab(vocab_path)
    functions = load_json_file(args.functions_definition)
    functions_parsed = parse_functions(functions)
    tests = load_json_file(args.input)
    tests_parsed = parse_tests(tests)
    result = generate_function_calls(vocab, functions_parsed,
                                     tests_parsed, model)
    save_json_file(args.output, result)
except ValueError as e:
    print(e)
    sys.exit(1)

"""Command line entry point: uv run python -m src [ARGS="options"].

Reads the function definitions and the test prompts, loads the model
and its vocabulary, generates one function call per prompt with
constrained decoding and writes the results as JSON.

Options (all optional):
    --functions_definition: JSON file with the available functions
        (default: data/input/functions_definition.json).
    --input: JSON file with the prompts
        (default: data/input/function_calling_tests.json).
    --output: JSON file to write
        (default: data/output/function_calling_results.json).

An error loading the files, loading the model or saving the output
prints "Error: ..." to stderr and exits with status 1. A failure in
a single prompt does not stop the run: that prompt gets an ERROR
entry.
"""

import argparse
import sys
import time
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

try:
    functions = load_json_file(args.functions_definition)
    functions_parsed = parse_functions(functions)
    tests = load_json_file(args.input)
    tests_parsed = parse_tests(tests)
    try:
        model = Small_LLM_Model()
        vocab_path = model.get_path_to_vocab_file()
    except Exception as e:
        raise ValueError(f"Could not load the model: {e}") from e
    vocab = load_vocab(vocab_path)

    start_time = time.time()
    result = generate_function_calls(vocab, functions_parsed,
                                     tests_parsed, model)
    elapsed = time.time() - start_time
    print(f"Generation took {elapsed:.2f} seconds")

    save_json_file(args.output, result)
except Exception as e:
    print(f"Error: {e}", file=sys.stderr)
    sys.exit(1)

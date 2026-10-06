*This project has been created as part of the 42 curriculum by agomez-a.*

# Call Me Maybe

Function calling for a small LLM (Qwen3-0.6B) using constrained decoding.

## Description

Large language models are good at understanding language but small ones
are unreliable at producing structured output: asked for JSON, they often
break the format. This project translates a natural-language request into
a structured function call, without answering the request itself.

Given the request `What is the sum of 40 and 2?` the program does not
return `42`. It returns the function to call and its typed arguments:

```json
{"name": "fn_add_numbers", "parameters": {"a": 40.0, "b": 2.0}}
```

The key idea is **constrained decoding**: the program never relies on the
model to "spontaneously" write valid JSON. The JSON skeleton is written by
the program, and the model only fills in the variable parts (the function
name and each argument value), one token at a time, choosing only among
the tokens that keep the output valid and compatible with the schema of
the chosen function. The result is JSON that can always be parsed and
that always matches the function definitions.

### Overview of the program

1. Read the function definitions and the test prompts (JSON files).
2. Validate them with Pydantic and report clear errors if they are wrong.
3. Load Qwen3-0.6B and its vocabulary through the provided `llm_sdk`.
4. For every prompt, generate the function call with constrained decoding.
5. Write one result object per prompt to a JSON output file.

### Project structure

```
.
├── Makefile
├── README.md
├── pyproject.toml
├── uv.lock
├── llm_sdk/                      # provided wrapper around the model
├── data/
│   └── input/
│       ├── functions_definition.json
│       └── function_calling_tests.json
└── src/
    ├── __main__.py               # command line entry point
    └── call_me_maybe/
        ├── __init__.py
        ├── constrained_decoding.py  # token validators and logit masking
        ├── file_utils.py            # JSON reading and writing
        ├── generation.py            # generation pipeline
        └── models.py                # Pydantic models and input parsing
```

## Instructions

### Requirements

- Python 3.10 or later
- uv

### Installation

```
make install      # runs: uv sync
```

`uv sync` creates the virtual environment and installs `numpy`,
`pydantic` and the local `llm_sdk` package. The first run downloads the
Qwen3-0.6B model.

### Running

```
make run
```

By default it reads `data/input/functions_definition.json` and
`data/input/function_calling_tests.json`, and writes
`data/output/function_calling_results.json`. Custom paths can be given
with `ARGS`:

```
make run ARGS="--functions_definition data/input/functions_definition.json \
               --input data/input/function_calling_tests.json \
               --output data/output/function_calls.json"
```

### Makefile rules

| Rule          | What it does                                              |
|---------------|-----------------------------------------------------------|
| `install`     | Installs the dependencies (`uv sync`).                    |
| `run`         | Runs the program (`ARGS` passes the options).             |
| `debug`       | Runs the program under `pdb`.                             |
| `clean`       | Removes `__pycache__` folders and `.mypy_cache`.          |
| `lint`        | `flake8` and `mypy` with the flags required by the subject. |
| `lint-strict` | `flake8` and `mypy --strict`.                             |

### Input and output format

Function definitions (`functions_definition.json`):

```json
[
  {
    "name": "fn_add_numbers",
    "description": "Add two numbers together and return their sum.",
    "parameters": {"a": {"type": "number"}, "b": {"type": "number"}},
    "returns": {"type": "number"}
  }
]
```

Supported parameter types are `string`, `number`, `integer` and `boolean`.

Test prompts (`function_calling_tests.json`):

```json
[{"prompt": "What is the sum of 2 and 3?"}]
```

Output (`function_calling_results.json`), one object per prompt, in the
same order:

```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": {"a": 2.0, "b": 3.0}
  }
]
```

Input limits enforced by the validation: function name and description up
to 200 characters, up to 10 parameters per function, parameter names up
to 20 characters, prompts up to 200 characters.

## Algorithm explanation

### Background

At each step the model returns one **logit** per token of its
vocabulary. Normally the token with the highest score is chosen. Constrained
decoding changes the scores *before* choosing:

1. The model produces the logits for all the tokens.
2. For every token, the program checks whether appending it to the text
   generated so far keeps the output valid.
3. The logit of every invalid token is set to `-inf`.
4. The token with the highest remaining logit is chosen (argmax). Since
   invalid tokens are `-inf`, they can never be selected.

To know the text of each token, the program loads the vocabulary file
(`get_path_to_vocab_file`) and inverts it (token id to token string).
Strings in this file are in byte-level BPE form, not plain text: for
example a leading space is written as `Ġ`.

### Building one function call

The answer for each prompt is built in this order:

```
{"name": "<function>", "parameters": {"<p1>": <v1>, "<p2>": <v2>}}
```

| Part                                   | Who writes it                         |
|----------------------------------------|---------------------------------------|
| `{"name": "`                           | Forced by the program                 |
| the function name                      | Chosen by the model (constrained)     |
| `", "parameters": {`                   | Forced by the program                 |
| each key (`"a":`) and the `, ` between | Forced by the program                 |
| each value                             | Chosen by the model (constrained)     |
| `}`                                    | Forced by the program                 |

Forced text is encoded with the model's tokenizer and appended to the list
of token ids, so the model "sees" it as if it had written it. The prompt
itself is a ChatML prompt (the format of Qwen) with the function
definitions in the system message and the request in the user message.

### Function name

At each step a token is allowed only if the text generated so far plus the
token is a **prefix of at least one function name**. The name is finished
when the decoded text is exactly one name and no other name starts with it.
The function is therefore always chosen by the model and is always one of
the existing functions.

If a name is a prefix of another one (for example `fn_add` and
`fn_add_numbers`), the text `fn_add` is a complete name but could still
grow. In that case the program compares the best logit among the tokens
that start with a closing quote against the best logit among the allowed
continuations, and the model decides whether to stop or to continue.

### Values by type

**string**: the program appends the opening quote, and then a token is
allowed if the content stays a valid JSON string: backslash escapes are
checked (`\"`, `\\`, `\n`...) and the first unescaped double quote means
the string ends. When the chosen token contains the closing quote, only the
text before it is kept. The closing quote is then appended and the content
is unescaped with `json.loads`, so a value such as `C:\\Users\\john` ends
up as `C:\Users\john`.

**number / integer**: a token is allowed if the text stays a valid number:
only digits, a `-` as the first character and, for `number`, a single `.`.
One leading space is ignored. At each step, if the token the model prefers
keeps the number valid it is accepted, and otherwise the number ends (the
model wants to write `,` or `}`). Until the first digit has been generated
the number cannot end: the best valid token is taken instead. A `number`
becomes a Python `float` and an `integer` an `int`.

**boolean**: a token is allowed if the text stays a prefix of `true` or
`false`, and the value ends when the decoded text is one of them.

### Why the output is always valid

- The structure (braces, keys, quotes, commas) is written by the program,
  not by the model.
- The keys are taken from the definition of the chosen function, in order,
  so no parameter is missing or extra.
- Every value can only be built from tokens accepted by its validator, and
  is converted to the Python type of the schema before being stored.
- The final object is built with a Pydantic model and written with
  `json.dump`.

## Design decisions

- **The program writes the skeleton, the model writes the values.** It
  guarantees valid JSON and a correct schema without relying on the prompt.
- **The function is chosen by the LLM**, not by keywords or heuristics, as
  required. Only the set of reachable names is restricted.
- **Validators are small pure functions** (`is_valid_prefix`,
  `is_valid_number_char`, `is_valid_string`) that receive the candidate text
  and return whether it is valid. The same `mask_logits` function is used for
  names, strings, numbers and booleans, only the validator changes.
- **Only the public methods of `llm_sdk` are used** (`encode`, `decode`,
  `get_logits_from_input_ids`, `get_path_to_vocab_file`).
- **Pydantic** validates the input files and builds the output objects.
- **One output entry per prompt.** If a prompt fails, its entry has the name
  `ERROR` and empty parameters, a warning is printed to stderr and the rest
  of the prompts are still processed. A blank prompt is treated the same way
  without calling the model.
- **Error handling.** Reading and parsing errors are raised as `ValueError`
  with the file name or the offending item in the message. `__main__.py`
  prints `Error: ...` to stderr and exits with status 1. Each prompt is also
  processed inside its own `try/except`, so one unexpected error cannot stop
  the run.
- **Token limits as a safety net**: 200 tokens for a function name and for a
  string, 20 for a number and for a boolean. They prevent endless generation
  when the model repeats itself. A string longer than 200 tokens is cut.
- **The `returns` field** is validated and included in the prompt but it is
  not used by the decoding.

## Performance analysis

### Accuracy

In manual testing across a variety of prompts and function definitions, the
vast majority of calls are correct. The failures observed are never format
errors — every output is valid JSON matching the chosen function's schema —
but cases where the model picks the wrong function, misreads an argument,
or mis-copies part of a value from the prompt. For example, a prompt whose
expected parameter value is an absolute file path (e.g. "/home/user/data.json")
can sometimes get a stray leading space inserted before the path instead of
being copied exactly.
This is a model limitation, not a decoding one: the decoding guarantees the
format, but the choice and copying of values depends on a 0.6B parameter model,
and prompts that need complex regular expressions are the hardest for it.
Prompting improvements reduced several of these errors (negative numbers,
missing punctuation in copied values) but could not fully eliminate them for
every pattern.

### Reliability

Every output is valid JSON and matches the schema of the chosen function,
because it is built from forced text and validated tokens only. Missing
files, invalid JSON and wrong input shapes produce a clear message and exit
status 1. Failures in a single prompt produce an `ERROR` entry and do not
stop the program.

### Speed

For a batch of around ten prompts, generation typically takes a few minutes, which is
close to the 5-minute limit the subject asks for. The exact time varies noticeably
between runs: it depends on how many string parameters are being generated, how long the
input prompts are (up to the 200-character limit enforced by validation), and how many
tokens the model needs before reaching a valid stopping point for each value.
Runs with several long strings or complex regex parameters have taken noticeably longer,
occasionally exceeding the target.
Hardware also matters — runs on a personal laptop are consistently faster than on the
school's evaluation machines.

The main cost is the masking step: for every generated token, the program runs the validator
over every token of the vocabulary in pure Python, plus one call to the model. The time
therefore grows with the number of generated tokens, and long strings are the most
expensive values.

## Challenges faced

- **Raw vocabulary tokens are not plain text.** A token such as `Ġ"` has
  `Ġ` for the space. Re-encoding the raw token string put a literal `Ġ` in
  the value. It was solved by decoding the chosen token with the model
  before using its text.
- **Negative numbers.** The space after `"a":` was forced by the program, but
  a space followed by `-` is a single token (`Ġ-`) that the model could no
  longer choose, so `-5` came out as `5`. It was solved by not forcing the
  space for numbers, ignoring one leading space in the number validator, and
  not allowing a number to end before its first digit.
- **Function names that are a prefix of another.** With plain prefix masking
  the shorter name could never be selected. It was solved by letting the
  model choose between closing the name and continuing it.
- **Backslashes and quotes inside strings.** JSON requires escaping, so the
  string validator tracks escape state, and the final value is unescaped
  with `json.loads`. Windows paths and quoted text are covered by the tests.
- **Model repeating itself.** On some prompts (for example regular
  expressions) the model tended to repeat the same pattern. The token limits
  stop the repetition from running forever.
- **Speed.** Masking the whole vocabulary at every step is slow in Python.

## Testing strategy

- Batteries of prompts that check that each result has the expected function
  and the expected arguments with the correct types. The prompts cover:
  addition, products and square roots, greetings, string reversal, regular
  expression substitution, a boolean function, SQL queries, reading files,
  Windows paths with backslashes, templates with quotes and braces, a very
  large decimal number, negative numbers (including `--5`) and functions
  whose names are a prefix of another (`fn_add` / `fn_add_numbers`).
- Edge cases from the subject: empty prompts, special characters, large
  numbers and ambiguous prompts.
- Static checks: `make lint` and `make lint-strict` (flake8 and mypy).

## Examples

Example prompts and results:

| Prompt                                  | Result                                                         |
|-----------------------------------------|----------------------------------------------------------------|
| `What is the sum of 2 and 3?`           | `fn_add_numbers`, `{"a": 2.0, "b": 3.0}`                       |
| `What is the sum of -5 and 3?`          | `fn_add_numbers`, `{"a": -5.0, "b": 3.0}`                      |
| `Greet shrek`                           | `fn_greet`, `{"name": "shrek"}`                                |
| `Reverse the string 'hello'`            | `fn_reverse_string`, `{"s": "hello"}`                          |

Error examples:

```
$ make run ARGS="--input data/input/missing.json"
Error: File not found: data/input/missing.json
```

If one prompt cannot be processed, a warning is printed to stderr, its entry
in the output is `{"prompt": "...", "name": "ERROR", "parameters": {}}` and
the other prompts are processed normally.

## Resources

### References

- Qwen3-0.6B model card: https://huggingface.co/Qwen/Qwen3-0.6B
- Constrained decoding: https://zeroentropy.dev/concepts/constrained-decoding/
- JSON specification (RFC 8259): https://www.rfc-editor.org/rfc/rfc8259
- Pydantic documentation: https://docs.pydantic.dev
- uv documentation: https://docs.astral.sh/uv/
- PEP 257, docstring conventions: https://peps.python.org/pep-0257/

### How AI was used

I used an Claude and Gemini for:

- Understanding concepts: how logits, tokenization and constrained decoding
  work, and how byte-level BPE tokens differ from plain text.
- Reviewing my code: fixing the `mypy` and `flake8` errors and checking the
  error handling and the error messages.
- Debugging specific problems: negative numbers, the literal `Ġ` characters in
  values, function names that are a prefix of another one, etc. In each case
  I got an explanation of what the problem could be and I tried applying and 
  testing the changes myself until I got it right.
- Writing the docstrings and drafting this README, which I reviewed against
  my final code.

"""Pydantic models and parsing of the input files."""

import pydantic
from pydantic import BaseModel, field_validator, Field
from typing import Literal, Any


class ParameterType(BaseModel):
    """Type of a parameter or of a return value.

    Attributes:
        type: One of "string", "number", "integer" or "boolean".
    """

    type: Literal["string", "number", "integer", "boolean"]


class FunctionDefinition(BaseModel):
    """Definition of a callable function, as read from the input file.

    Attributes:
        name: Function name (max 200 characters).
        description: What the function does (max 200 characters). It
            is part of the prompt and helps the model choose.
        parameters: Parameter name -> type (max 10 parameters, names
            up to 20 characters).
        returns: Return type. It is included in the prompt but not
            used by the decoding.
    """

    name: str = Field(max_length=200)
    description: str = Field(max_length=200)
    parameters: dict[str, ParameterType] = Field(max_length=10)
    returns: ParameterType

    @field_validator("parameters")
    @classmethod
    def is_param_length_valid(cls,
                              v: dict[str,
                                      ParameterType]) -> dict[str,
                                                              ParameterType]:
        """Rejects parameter names longer than 20 characters.

        Args:
            v: Parameters dict being validated.

        Returns:
            The same dict, unchanged.

        Raises:
            ValueError: If a parameter name is longer than 20
                characters (Pydantic wraps it in a ValidationError).
        """

        for param_name in v.keys():
            if len(param_name) > 20:
                raise ValueError(
                    f"Parameter name '{param_name[:30]}...' is too long. "
                    f"Max 20 characters.")
        return v


class TestPrompt(BaseModel):
    """A natural-language request to translate into a function call.

    Blank prompts are accepted here and handled later: they produce an
    ERROR result instead of stopping the program.

    Attributes:
        prompt: Request text (max 200 characters).
    """

    prompt: str = Field(max_length=200)


class FunctionCallResult(BaseModel):
    """One entry of the output file.

    Attributes:
        prompt: The original request.
        name: Name of the chosen function ("ERROR" if generation
            failed).
        parameters: Parameter name -> generated value (str, int,
            float or bool). Empty if generation failed.
    """

    prompt: str
    name: str
    parameters: dict[str, str | int | float | bool]


def parse_functions(
        functions: list[dict[str, Any]]) -> list[FunctionDefinition]:
    """Validates the content of the functions definition file.

    Args:
        functions: Parsed JSON content of the file.

    Returns:
        List of validated function definitions.

    Raises:
        ValueError: If the content is not a JSON array, an item is
            not a JSON object, an item does not match the schema, or
            the array is empty.
    """

    if not isinstance(functions, list):
        raise ValueError(
            "functions_definition.json must contain a JSON array")
    result = []
    for func in functions:
        if not isinstance(func, dict):
            raise ValueError(
                f"Each function must be a JSON object, got: {func}")
        try:
            function_def = FunctionDefinition.model_validate(func)
            result.append(function_def)
        except pydantic.ValidationError as e:
            msg = format_functions_validation_error(
                e, func.get("name", "unknown"))
            raise ValueError(msg) from e
    if not result:
        raise ValueError("functions_definition.json must define "
                         "at least one function")
    return result


def parse_tests(tests: list[dict[str, Any]]) -> list[TestPrompt]:
    """Validates the content of the test prompts file.

    Args:
        tests: Parsed JSON content of the file.

    Returns:
        List of validated test prompts.

    Raises:
        ValueError: If the content is not a JSON array, an item is
            not a JSON object, or an item does not match the schema.
    """

    if not isinstance(tests, list):
        raise ValueError(
            "function_calling_tests.json must contain a JSON array")
    result = []
    for test in tests:
        if not isinstance(test, dict):
            raise ValueError(
                f"Each test must be a JSON object, got: {test}")
        try:
            test_def = TestPrompt.model_validate(test)
            result.append(test_def)
        except pydantic.ValidationError as e:
            msg = format_tests_validation_error(
                e, test.get("prompt", "unknown"))
            raise ValueError(msg) from e
    return result


def format_functions_validation_error(e: pydantic.ValidationError,
                                      function_name: str) -> str:
    """Builds a readable message from a Pydantic validation error.

    One line per error, with the form:
    Invalid function '<name>', field "<path>": <reason>

    Args:
        e: Validation error raised by Pydantic.
        function_name: Name of the function being validated, used in
            the message.

    Returns:
        The lines joined with newlines.
    """

    messages = []
    for error in e.errors():
        field_path = ".".join(str(loc) for loc in error["loc"])
        reason = error["msg"]
        location = f', field "{field_path}"' if field_path else ''
        error_msg = (f"Invalid function '{function_name}'{location}"
                     f": {reason}")
        messages.append(error_msg)
    return "\n".join(messages)


def format_tests_validation_error(e: pydantic.ValidationError,
                                  prompt: str) -> str:
    """Builds a readable message from a Pydantic validation error.

    One line per error, with the form:
    Invalid test prompt '<prompt>', field "<path>": <reason>

    Args:
        e: Validation error raised by Pydantic.
        prompt: Prompt being validated, used in the message.

    Returns:
        The lines joined with newlines.
    """

    messages = []
    for error in e.errors():
        field_path = ".".join(str(loc) for loc in error["loc"])
        reason = error["msg"]
        location = f', field "{field_path}"' if field_path else ''
        error_msg = (f"Invalid test prompt '{prompt}'{location}"
                     f": {reason}")
        messages.append(error_msg)
    return "\n".join(messages)

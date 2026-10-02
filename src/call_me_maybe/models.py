import pydantic
from pydantic import BaseModel, field_validator, Field
from typing import Literal, Any


class ParameterType(BaseModel):
    type: Literal["string", "number", "integer", "boolean"]


class FunctionDefinition(BaseModel):
    name: str = Field(max_length=200)
    description: str = Field(max_length=200)
    parameters: dict[str, ParameterType] = Field(max_items=10)
    returns: ParameterType

    @field_validator("parameters")
    @classmethod
    def is_param_length_valid(cls, v: dict[str, ParameterType]):
        for param_name in v.keys():
            if len(param_name) > 20:
                raise ValueError(
                    f"Parameter name '{param_name[:30]}...' is too long. "
                    f"Max 20 characters.")
        return v


class TestPrompt(BaseModel):
    prompt: str = Field(max_length=200)

    @field_validator("prompt")
    @classmethod
    def prompt_must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("prompt must not be empty or blank")
        return v


class FunctionCallResult(BaseModel):
    prompt: str
    name: str
    parameters: dict[str, str | int | float | bool]


def parse_functions(
        functions: list[dict[str, Any]]) -> list[FunctionDefinition]:
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
    return result


def parse_tests(tests: list[dict[str, Any]]) -> list[TestPrompt]:
    if not isinstance(tests, list):
        raise ValueError(
            "functions_definition.json must contain a JSON array")
    result = []
    for test in tests:
        if not isinstance(test, dict):
            raise ValueError(
                f"Each function must be a JSON object, got: {test}")
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
    messages = []
    for error in e.errors():
        field_path = ".".join(str(loc) for loc in error["loc"])
        reason = error["msg"]
        error_msg = (f"Invalid function '{function_name}'"
                     f"{f', field \"{field_path}\"' if field_path else ''}"
                     f": {reason}")
        messages.append(error_msg)
    return "\n".join(messages)


def format_tests_validation_error(e: pydantic.ValidationError,
                                  prompt: str) -> str:
    messages = []
    for error in e.errors():
        field_path = ".".join(str(loc) for loc in error["loc"])
        reason = error["msg"]
        error_msg = (f"Invalid test prompt '{prompt}'"
                     f"{f', field \"{field_path}\"' if field_path else ''}"
                     f": {reason}")
        messages.append(error_msg)
    return "\n".join(messages)

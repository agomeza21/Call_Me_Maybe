import pydantic
from pydantic import BaseModel
from typing import Literal


class ParameterType(BaseModel):
    type: Literal["string", "number", "integer", "boolean"]


class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: dict[str, ParameterType]
    returns: ParameterType


class TestPrompt(BaseModel):
    prompt: str


def parse_functions(functions: list[dict]) -> list[FunctionDefinition]:
    result = []
    for func in functions:
        try:
            function_def = FunctionDefinition.model_validate(func)
            result.append(function_def)
        except pydantic.ValidationError as e:
            msg = format_functions_validation_error(
                e, func.get("name", "unknown"))
            raise ValueError(msg) from e
    return result


def parse_tests(tests: list[dict]) -> list[TestPrompt]:
    result = []
    for test in tests:
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
        value = error["input"]
        expected = error["ctx"]["expected"]
        error_msg = (f"Invalid type '{value}' in function '{function_name}':"
                     f" expected one of {expected}")
        messages.append(error_msg)
    return "\n".join(messages)


def format_tests_validation_error(e: pydantic.ValidationError,
                                  prompt: str) -> str:
    messages = []
    for error in e.errors():
        value = error["input"]
        expected = error["ctx"]["expected"]
        error_msg = (f"Invalid type '{value}' in prompt '{prompt}':"
                     f" expected one of {expected}")
        messages.append(error_msg)
    return "\n".join(messages)

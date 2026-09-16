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
            msg = format_validation_error(e, func["name"])
            raise ValueError(msg) from e
    return result


def parse_tests(tests: list[dict]) -> list[TestPrompt]:
    result = []
    for test in tests:
        test_def = TestPrompt.model_validate(test)
        result.append(test_def)
    return result


def format_validation_error(e: pydantic.ValidationError,
                            function_name: str) -> str:
    messages = []
    for error in e.errors():
        value = error["input"]
        name = error["loc"][-2]
        expected = error["ctx"]["expected"]
        error_msg = (f"Invalid type '{value}' for parameter '{name}' in "
                     f"function '{function_name}': expected one of {expected}")
        messages.append(error_msg)
    return "\n".join(messages)

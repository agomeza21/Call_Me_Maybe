from call_me_maybe.models import FunctionDefinition
import pydantic

bad_func = {
    "name": "fn_test",
    "description": "test",
    "parameters": {},
    "returns": {"type": "banana"}
}

try:
    FunctionDefinition.model_validate(bad_func)
except pydantic.ValidationError as e:
    print(e.errors())

"""Argument checking shared by the tool CLIs. Each tool gets its JSON arguments as one argv string, checks them
against its own schema and reports problems as the tool result, in pi's message format."""

import json


def validate(raw_args, tool, error):
    """Parses the JSON arguments and checks them against `tool`'s schema (an OpenAI function tool dict). Returns
    the kwargs for the tool function. Bad arguments raise `error` (the tool's own exception class) with the message
    the model sees. No type conversion. Unknown keys are ignored, like in pi; this also drops the `why_this_tool`
    and `why_not_<tool>` keys that the reason prompts add."""
    try:
        args = json.loads(raw_args)
    except ValueError as e:
        raise error(f"Error parsing tool call arguments: {e}.") from e
    if not isinstance(args, dict):
        raise error("Tool call arguments must be a JSON object.")
    name = tool["function"]["name"]
    params = tool["function"]["parameters"]
    props = params["properties"]
    types = {"string": str, "boolean": bool, "number": (int, float)}
    errors = [f"  - /{key}: Expected required property" for key in params.get("required", []) if key not in args]
    for key, value in args.items():
        if key in props and not isinstance(value, types[props[key]["type"]]):
            errors.append(f"  - /{key}: Expected {props[key]['type']}")
    if errors:
        raise error(
            f'Validation failed for tool "{name}":\n' + "\n".join(errors)
            + f"\n\nReceived arguments:\n{json.dumps(args, indent=2)}"
        )
    return {key: value for key, value in args.items() if key in props}

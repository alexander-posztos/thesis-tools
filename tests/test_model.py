import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from minisweagent.exceptions import FormatError

from minisweagent.models.utils.actions_toolcall import BASH_TOOL

from thesis_tools.model import BashModel, ToolsModel
from thesis_tools.search import SEARCH_TOOL

ROOT = Path(__file__).resolve().parent.parent


def test_tool_surface():
    assert [t["function"]["name"] for t in BashModel(model_name="x").tools] == ["bash"]
    assert [t["function"]["name"] for t in ToolsModel(model_name="x").tools] == [
        "bash",
        "search",
    ]


def test_tool_directive():
    free = ToolsModel(model_name="x").tools[1]["function"]["description"]
    assert free == SEARCH_TOOL["function"]["description"]
    steered = ToolsModel(model_name="x", tool_directive="Use me. ").tools[1][
        "function"
    ]["description"]
    assert steered == "Use me. " + free
    assert (
        SEARCH_TOOL["function"]["description"] == free
    )  # the module constant is untouched


def test_reason_prompt():
    # without the prompt the schemas are the module constants, nothing is added
    assert BashModel(model_name="x").tools == (BASH_TOOL,)
    assert ToolsModel(model_name="x").tools[1] is SEARCH_TOOL

    for tool in ToolsModel(model_name="x", reason_prompt=" Why? ").tools:
        params = tool["function"]["parameters"]
        assert list(params["properties"])[0] == "why_this_tool"  # first, so the model writes it before the action
        assert params["properties"]["why_this_tool"] == {"type": "string", "description": "Why?"}
        assert params["required"][0] == "why_this_tool"
    bash = BashModel(model_name="x", reason_prompt="Why?").tools[0]["function"]["parameters"]
    assert bash["required"] == ["why_this_tool", "command"]
    # a single-tool surface asks the single-tool question, a multi-tool surface the comparison
    kw = dict(model_name="x", reason_prompt="Over others?", reason_prompt_single="Alone?")
    assert BashModel(**kw).tools[0]["function"]["parameters"]["properties"]["why_this_tool"]["description"] == "Alone?"
    for tool in ToolsModel(**kw).tools:
        assert tool["function"]["parameters"]["properties"]["why_this_tool"]["description"] == "Over others?"
    # the module constants are untouched
    assert "why_this_tool" not in BASH_TOOL["function"]["parameters"]["properties"]
    assert "why_this_tool" not in SEARCH_TOOL["function"]["parameters"]["properties"]


def response(*calls):
    """A litellm-shaped response with one tool call per (name, args) pair."""
    tool_calls = [
        SimpleNamespace(id=f"c{i}", function=SimpleNamespace(name=name, arguments=json.dumps(args)))
        for i, (name, args) in enumerate(calls)
    ]
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=tool_calls), finish_reason="stop")])


def test_missing_reason_is_a_format_error():
    kw = dict(model_name="x", reason_prompt="Why?", format_error_template="bad: {{error}}")
    good = {"why_this_tool": "bash: fits.", "command": "ls"}
    for model in (BashModel(**kw), ToolsModel(**kw)):
        assert model._parse_actions(response(("bash", good)))[0]["command"] == "ls"
        for args in ({"command": "ls"}, {"why_this_tool": " ", "command": "ls"}):
            with pytest.raises(FormatError) as e:
                model._parse_actions(response(("bash", good), ("bash", args)))
            assert e.value.messages[0]["content"] == "bad: Missing 'why_this_tool' argument in bash tool call."
    tools = ToolsModel(**kw)
    with pytest.raises(FormatError) as e:
        tools._parse_actions(response(("search", {"pattern": "x"})))
    assert "in search tool call" in e.value.messages[0]["content"]
    # without the prompt the field is not required
    assert BashModel(model_name="x")._parse_actions(response(("bash", {"command": "ls"})))[0]["command"] == "ls"


def test_minimal_yamls_share_the_prompt():
    base = yaml.safe_load((ROOT / "minimal.yaml").read_text())
    for name, key in [("minimal-directive.yaml", "tool_directive"), ("minimal-reason.yaml", "reason_prompt")]:
        other = yaml.safe_load((ROOT / name).read_text())
        if key == "reason_prompt":  # the reason arm adds exactly one paragraph about the field, nothing else
            lines = other["agent"]["instance_template"].splitlines()
            extra = [l for l in lines if "why_this_tool" in l]
            assert len(extra) == 1
            kept = "\n".join(l for l in lines if l not in extra).replace("\n\n\n", "\n\n") + "\n"
            assert kept == base["agent"]["instance_template"]
            assert {k: v for k, v in other["agent"].items() if k != "instance_template"} == {
                k: v for k, v in base["agent"].items() if k != "instance_template"}
        else:
            assert other["agent"] == base["agent"]
        assert other["model"]["format_error_template"] == base["model"]["format_error_template"]
        assert other["model"][key]

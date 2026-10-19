import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from minisweagent.exceptions import FormatError

from minisweagent.models.utils.actions_toolcall import BASH_TOOL

from thesis_tools.model import BashModel, ToolsModel, runs_grep
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
    # with reason_against_prompt every tool gets one why_not_<other> field after why_this_tool, a single tool none
    kw = dict(model_name="x", reason_prompt="Why?", reason_against_prompt="Why not {tool}?")
    assert list(BashModel(**kw).tools[0]["function"]["parameters"]["properties"]) == ["why_this_tool", "command"]
    bash, search = (t["function"]["parameters"] for t in ToolsModel(**kw).tools)
    assert bash["required"][:3] == ["why_this_tool", "why_not_search", "command"]
    assert bash["properties"]["why_not_search"]["description"] == "Why not search?"
    assert list(search["properties"])[:2] == ["why_this_tool", "why_not_bash"]
    assert search["required"][:2] == ["why_this_tool", "why_not_bash"]
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
    good = {"why_this_tool": "fits.", "command": "ls"}
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
    # the why_not_<other> fields are required too, and only the ones of the called tool
    tools = ToolsModel(reason_against_prompt="Why not {tool}?", **kw)
    with pytest.raises(FormatError) as e:
        tools._parse_actions(response(("bash", good)))
    assert e.value.messages[0]["content"] == "bad: Missing 'why_not_search' argument in bash tool call."
    assert tools._parse_actions(response(("bash", good | {"why_not_search": "no."})))[0]["command"] == "ls"
    with pytest.raises(FormatError):
        tools._parse_actions(response(("search", {"why_this_tool": "a.", "why_not_search": "b.", "pattern": "x"})))
    # without the prompt the field is not required
    assert BashModel(model_name="x")._parse_actions(response(("bash", {"command": "ls"})))[0]["command"] == "ls"


def test_runs_grep():
    blocked = ["grep -rn foo src", "cd /x && grep -n foo a.py | head", "rg foo", "git grep foo", "find . | xargs grep foo",
               "ls || grep foo", "sleep 5; grep -c BUILD /tmp/log", "egrep foo a"]
    allowed = ["ps aux | grep -c rustc", "git log --grep=foo", "cat a | grep foo | grep -v bar", "./rg foo", "my_grep x",
               "python3 -c 'import re; re.search(1,2)'"]
    assert [c for c in blocked if not runs_grep(c)] == []
    assert [c for c in allowed if runs_grep(c)] == []


def test_block_grep_is_a_format_error():
    kw = dict(model_name="x", block_grep=True, format_error_template="bad: {{error}}")
    model = ToolsModel(**kw)
    assert model._parse_actions(response(("bash", {"command": "ps aux | grep x"})))[0]["command"] == "ps aux | grep x"
    assert "thesis_tools.search" in model._parse_actions(response(("search", {"pattern": "grep"})))[0]["command"]
    with pytest.raises(FormatError) as e:
        model._parse_actions(response(("bash", {"command": "ls"}), ("bash", {"command": "grep -rn x ."})))
    assert e.value.messages[0]["content"].startswith("bad: grep and rg are not available")
    assert ToolsModel(model_name="x")._parse_actions(response(("bash", {"command": "grep -rn x ."})))[0]["command"]


def test_minimal_yamls_share_the_prompt():
    base = yaml.safe_load((ROOT / "minimal.yaml").read_text())
    for name, key in [("minimal-directive.yaml", "tool_directive"), ("minimal-reason.yaml", "reason_prompt"),
                      ("minimal-reason-nogrep.yaml", "block_grep")]:
        other = yaml.safe_load((ROOT / name).read_text())
        if key != "tool_directive":  # the reason arm adds exactly one paragraph about the field, the nogrep arm one more
            lines = other["agent"]["instance_template"].splitlines()
            extra = [l for l in lines if "why_this_tool" in l or "grep" in l]
            assert len(extra) == (1 if key == "reason_prompt" else 2)
            reason = yaml.safe_load((ROOT / "minimal-reason.yaml").read_text())["model"]
            assert all(other["model"][k] == reason[k] for k in ("reason_prompt", "reason_against_prompt"))
            kept = re.sub(r"\n{3,}", "\n\n", "\n".join(l for l in lines if l not in extra)) + "\n"
            assert kept == base["agent"]["instance_template"]
            assert {k: v for k, v in other["agent"].items() if k != "instance_template"} == {
                k: v for k, v in base["agent"].items() if k != "instance_template"}
        else:
            assert other["agent"] == base["agent"]
        assert other["model"]["format_error_template"] == base["model"]["format_error_template"]
        assert other["model"][key]

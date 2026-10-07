from pathlib import Path

import yaml

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


def test_minimal_yamls_share_the_prompt():
    a = yaml.safe_load((ROOT / "minimal.yaml").read_text())
    b = yaml.safe_load((ROOT / "minimal-directive.yaml").read_text())
    assert a["agent"] == b["agent"]
    assert a["model"]["format_error_template"] == b["model"]["format_error_template"]
    assert b["model"]["tool_directive"]

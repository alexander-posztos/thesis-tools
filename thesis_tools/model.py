"""Extending the LitellmModel from mini-swe-agent, because the model class sends every LLM request and the
tools list is part of that request. So this is where the tool surface is set.

BashModel is the baseline (bash only tool surface).
ToolsModel is the extension, currently only the search tool. We added tool_directive as a yaml key to easily be
able to edit the tool description without having to change the code. We can edit the yaml file directly on the VM,
but to edit search.py we would have to commit and push every time, since Pier installs the package from GitHub.

reason_prompt (both classes) adds a required `reason` string as the first parameter of every tool, so the model
explains why it chose the tool before it writes the arguments. The reason is stored along with the tool call
arguments in the trajectory for later analysis.
To not pollute or change the agent performance, we drop the reason string before it reaches the agent

Pier selects one with --ak model_class=thesis_tools.model.<class>. Both abort the trial on a context overflow.
"""

import copy
import shlex
import sys

import litellm
from minisweagent.models.litellm_model import LitellmModel, LitellmModelConfig
from minisweagent.models.utils.actions_toolcall import BASH_TOOL, parse_toolcall_actions

from thesis_tools.search import SEARCH_TOOL


class BashModelConfig(LitellmModelConfig):
    reason_prompt: str = ""
    """Description of a required `reason` parameter put first in every tool's schema. Empty means no parameter."""


def add_reason(tool, prompt):
    """Returns a copy of a tool schema with `reason` as its first, required parameter. Properties keep their order
    in the JSON schema, and models fill the arguments in that order, so the reason comes before the action.
    The original dict is untouched (BASH_TOOL and SEARCH_TOOL are module constants)."""
    if not prompt:
        return tool
    tool = copy.deepcopy(tool)
    params = tool["function"]["parameters"]
    params["properties"] = {"reason": {"type": "string", "description": prompt.strip()}} | params["properties"]
    params["required"] = ["reason"] + [k for k in params.get("required", []) if k != "reason"]
    return tool


class BashModel(LitellmModel):
    """Baseline. A copy of mini-swe-agent's LitellmModel with two changes.

    1. tools=[BASH_TOOL] is not hardcoded, we send self.tools instead, so ToolsModel can add tools.
    2. the except block immediately stops the trial on a context overflow. Aqueduct sends it as a plain
       BadRequestError, which mini retries until the rate limit or the timeout kills the trial.
       Re-raising it as ContextWindowExceededError makes mini abort right away.
    """

    def __init__(self, *, config_class=BashModelConfig, **kwargs):
        super().__init__(config_class=config_class, **kwargs)
        self.tools = tuple(add_reason(tool, self.config.reason_prompt) for tool in self._tools())

    def _tools(self):
        """The tool surface of this condition, before the reason parameter is added. Subclasses override."""
        return [BASH_TOOL]

    def _query(self, messages, **kwargs):
        try:
            return litellm.completion(
                model=self.config.model_name,
                messages=messages,
                tools=self.tools,
                **(self.config.model_kwargs | kwargs),
            )
        except litellm.exceptions.BadRequestError as e:
            if "maximum context length" in str(e) and not isinstance(e, litellm.exceptions.ContextWindowExceededError):
                raise litellm.exceptions.ContextWindowExceededError(
                    str(e), self.config.model_name, getattr(e, "llm_provider", "")
                ) from e
            raise


class ToolsModelConfig(BashModelConfig):
    tool_directive: str = ""
    """Sentence put in front of the search tool description. Empty means nothing is prepended."""


class ToolsModel(BashModel):
    """Extension. Adds our search tool next to bash. A search call becomes a shell command that runs
    thesis_tools.search in the container, so environment, observations and trajectories stay as in mini-swe-agent."""

    def __init__(self, **kwargs):
        super().__init__(config_class=ToolsModelConfig, **kwargs)

    def _tools(self):
        search_tool = SEARCH_TOOL
        if self.config.tool_directive:
            search_tool = copy.deepcopy(SEARCH_TOOL)
            search_tool["function"]["description"] = (
                self.config.tool_directive.strip() + " " + search_tool["function"]["description"]
            )
        return [BASH_TOOL, search_tool]

    def _parse_actions(self, response):
        tool_calls = response.choices[0].message.tool_calls or []
        if not tool_calls:
            return super()._parse_actions(response)  # raises mini-swe-agent's "no tool calls" FormatError
        actions = []
        for tool_call in tool_calls:
            if tool_call.function.name == "search":
                # the arguments go to search.py as they are, it checks them and reports problems as the tool result
                command = shlex.join([sys.executable, "-m", "thesis_tools.search", tool_call.function.arguments])
                actions.append({"command": command, "tool_call_id": tool_call.id})
            else:
                # bash (and unknown names) get mini's own validation and error messages
                actions += parse_toolcall_actions(
                    [tool_call],
                    format_error_template=self.config.format_error_template,
                    template_kwargs={"finish_reason": response.choices[0].finish_reason},
                )
        return actions

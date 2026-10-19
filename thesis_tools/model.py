"""Extending the LitellmModel from mini-swe-agent, because the model class sends every LLM request and the
tools list is part of that request. So this is where the tool surface is set.

BashModel is the baseline (bash only tool surface).
ToolsModel is the extension, currently only the search tool. We added tool_directive as a yaml key to easily be
able to edit the tool description without having to change the code. We can edit the yaml file directly on the VM,
but to edit search.py we would have to commit and push every time, since Pier installs the package from GitHub.

reason_prompt (both classes) adds a required `why_this_tool` string as the first parameter of every tool, so the
model explains why it chose the tool before it writes the arguments. reason_against_prompt adds one required
`why_not_<tool>` string per other tool right after it (none on a single-tool surface). The fields are stored along
with the tool call arguments in the trajectory for later analysis. Until 19 Oct 2026 a single field asked for
"<chosen tool>: <reason>. <other tool>: <reason against>.", and GLM 5.3 swapped the labels in 73 of 547 coded calls
(a bash grep explained under "search:"), in streaks within a trial; with the tool names in the field names the
model cannot swap them. The field was called `reason` until 18 Oct 2026; with that name the models wrote step
labels ("Find X definition") whatever the description said. A call without a field gets a format error (GLM 5.3
left it out of most search calls although the schema marks it required). The reason strings never reach the
agent, so they cannot change the agent's behaviour.

block_grep (ToolsModel only) is the leave-one-out arm: a bash command that runs grep or rg as a command gets a
format error that points to the search tool, so the search tool is the only way to search file contents. grep as
a filter after a pipe stays allowed, since those calls (ps aux | grep, git log | grep) are not searches.

Pier selects one with --ak model_class=thesis_tools.model.<class>. Both abort the trial on a context overflow.
"""

import copy
import json
import re
import shlex
import sys

import litellm
from jinja2 import StrictUndefined, Template
from minisweagent.exceptions import FormatError
from minisweagent.models.litellm_model import LitellmModel, LitellmModelConfig
from minisweagent.models.utils.actions_toolcall import BASH_TOOL, parse_toolcall_actions

from thesis_tools.search import SEARCH_TOOL


class BashModelConfig(LitellmModelConfig):
    reason_prompt: str = ""
    """Description of a required `why_this_tool` parameter put first in every tool's schema. Empty means no parameter."""
    reason_against_prompt: str = ""
    """Description of a required `why_not_<tool>` parameter per other tool, after `why_this_tool`. `{tool}` is
    replaced by the other tool's name. Only used together with reason_prompt."""


REASON_FIELD = "why_this_tool"


def reason_fields(names, prompt, against):
    """Maps each tool name to its reason parameters (field name -> description): REASON_FIELD, then why_not_<other>
    per other tool if `against` is set. Empty dicts without a prompt."""
    fields = {}
    for name in names:
        fields[name] = {REASON_FIELD: prompt.strip()} if prompt else {}
        if prompt and against:
            fields[name] |= {f"why_not_{o}": against.strip().replace("{tool}", o) for o in names if o != name}
    return fields


def add_reasons(tool, fields):
    """Returns a copy of a tool schema with `fields` as its first, required parameters. Properties keep their order
    in the JSON schema, and models fill the arguments in that order, so the reasons come before the action.
    The original dict is untouched (BASH_TOOL and SEARCH_TOOL are module constants)."""
    if not fields:
        return tool
    tool = copy.deepcopy(tool)
    params = tool["function"]["parameters"]
    params["properties"] = {k: {"type": "string", "description": v} for k, v in fields.items()} | params["properties"]
    params["required"] = list(fields) + [k for k in params.get("required", []) if k not in fields]
    return tool


def check_reasons(tool_calls, fields, format_error_template, finish_reason):
    """Raises mini-swe-agent's FormatError (same template and message shape as parse_toolcall_actions) for the first
    tool call that leaves one of its reason fields (`fields` from reason_fields) missing or blank. Unparseable
    arguments and unknown tool names are left to the per-tool validation."""
    for tool_call in tool_calls:
        try:
            args = json.loads(tool_call.function.arguments)
        except Exception:
            continue
        if not isinstance(args, dict):
            continue
        for field in fields.get(tool_call.function.name, {}):
            if not str(args.get(field, "")).strip():
                raise FormatError(
                    {
                        "role": "user",
                        "content": Template(format_error_template, undefined=StrictUndefined).render(
                            actions=[],
                            error=f"Missing '{field}' argument in {tool_call.function.name} tool call.",
                            has_tool_calls=True,
                            finish_reason=finish_reason,
                        ),
                        "extra": {"interrupt_type": "FormatError"},
                    }
                )


class BashModel(LitellmModel):
    """Baseline. A copy of mini-swe-agent's LitellmModel with two changes.

    1. tools=[BASH_TOOL] is not hardcoded, we send self.tools instead, so ToolsModel can add tools.
    2. the except block immediately stops the trial on a context overflow. Aqueduct sends it as a plain
       BadRequestError, which mini retries until the rate limit or the timeout kills the trial.
       Re-raising it as ContextWindowExceededError makes mini abort right away.
    """

    def __init__(self, *, config_class=BashModelConfig, **kwargs):
        super().__init__(config_class=config_class, **kwargs)
        tools = self._tools()
        names = [tool["function"]["name"] for tool in tools]
        self.reason_fields = reason_fields(names, self.config.reason_prompt, self.config.reason_against_prompt)
        self.tools = tuple(add_reasons(tool, self.reason_fields[name]) for tool, name in zip(tools, names))

    def _tools(self):
        """The tool surface of this condition, before the reason parameter is added. Subclasses override."""
        return [BASH_TOOL]

    def _parse_actions(self, response):
        check_reasons(
            response.choices[0].message.tool_calls or [],
            self.reason_fields,
            self.config.format_error_template,
            response.choices[0].finish_reason,
        )
        return super()._parse_actions(response)

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
    block_grep: bool = False
    """Reject bash commands that run grep or rg as a command (not as a filter after a pipe)."""


# grep, rg, egrep or fgrep as a command word, not part of a longer token (--grep=, ./rg, my_grep) and not
# directly after a single pipe. `git grep`, `xargs grep`, `cd x && grep` and `find | xargs grep` are all blocked.
GREP_COMMAND = re.compile(r"(?<![\w./-])(?:grep|rg|egrep|fgrep)\b")
BLOCKED_GREP = "grep and rg are not available as commands in this run, only as a filter after a pipe. Use the search tool to find text in files."


def runs_grep(command):
    """True if the bash command runs grep/rg as a command. A match right after a single `|` is a filter and allowed."""
    for m in GREP_COMMAND.finditer(command):
        before = command[: m.start()].rstrip()
        if before.endswith("|") and not before.endswith("||"):
            continue
        return True
    return False


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
        check_reasons(tool_calls, self.reason_fields, self.config.format_error_template, response.choices[0].finish_reason)
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
        if self.config.block_grep:
            for action in actions:
                if "thesis_tools.search" not in action["command"] and runs_grep(action["command"]):
                    raise FormatError(
                        {
                            "role": "user",
                            "content": Template(self.config.format_error_template, undefined=StrictUndefined).render(
                                actions=[], error=BLOCKED_GREP, has_tool_calls=True,
                                finish_reason=response.choices[0].finish_reason,
                            ),
                            "extra": {"interrupt_type": "FormatError"},
                        }
                    )
        return actions

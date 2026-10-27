"""Extending the LitellmModel from mini-swe-agent, because the model class sends every LLM request and the
tools list is part of that request. So this is where the tool surface is set.

BashModel is the baseline (bash only tool surface).
ToolsModel is the extension, it has bash, plus the named tools listed in the `tools` yaml key 
(currently search and/or read). We added tool_directive as a yaml key to easily be able to edit the search tool description without having to change the
code. We can edit the yaml file directly on the VM, but to edit search.py we would have to commit and push every
time, since Pier installs the package from GitHub.

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

block_grep and block_read (ToolsModel only) are the leave-one-out functionallity. When we want to "force" the agent
to use the named tools instead of bash, we check whether the submitted  bash command runs grep or rg
(for block_grep) or cat, head, nl or sed -n (for block_read), if this is the case, the command gets a format error that points to the named
tool, so the named tool is the only way to search or read file contents. The same commands as a filter after a
pipe stay allowed, since those calls (ps aux | grep, git log | head) are not searches or file reads, and heredoc
bodies are not looked at (a Python patch script that mentions grep is not a grep call).

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

from thesis_tools.read import READ_TOOL
from thesis_tools.search import SEARCH_TOOL

NAMED_TOOLS = {"search": (SEARCH_TOOL, "thesis_tools.search"), "read": (READ_TOOL, "thesis_tools.read")}

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


def format_error(template, error, finish_reason):
    """mini-swe-agent's FormatError with the same template and message shape as parse_toolcall_actions."""
    return FormatError(
        {
            "role": "user",
            "content": Template(template, undefined=StrictUndefined).render(
                actions=[], error=error, has_tool_calls=True, finish_reason=finish_reason
            ),
            "extra": {"interrupt_type": "FormatError"},
        }
    )


def check_reasons(tool_calls, fields, format_error_template, finish_reason):
    """Raises a FormatError for the first tool call that leaves one of its reason fields (`fields` from
    reason_fields) missing or blank. Unparseable arguments and unknown tool names are left to the per-tool
    validation."""
    for tool_call in tool_calls:
        try:
            args = json.loads(tool_call.function.arguments)
        except Exception:
            continue
        if not isinstance(args, dict):
            continue
        for field in fields.get(tool_call.function.name, {}):
            if not str(args.get(field, "")).strip():
                error = f"Missing '{field}' argument in {tool_call.function.name} tool call."
                raise format_error(format_error_template, error, finish_reason)


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
    tools: list[str] = ["search"]
    """The named tools, in this order. Names are defined in NAMED_TOOLS."""
    tool_directive: str = ""
    """String put in front of the search tool description (to optionally nudge the model to use this tool). Empty means nothing is prepended."""
    block_grep: bool = False
    """Reject bash commands that run grep or rg as a command (not as a filter after a pipe)."""
    block_read: bool = False
    """Reject bash commands that run cat, head, nl or sed -n as a command (not as a filter after a pipe)."""


# grep, rg, egrep or fgrep as a command word, not part of a longer token (--grep=, ./rg, my_grep) and not
# directly after a single pipe. `git grep`, `xargs grep`, `cd x && grep` and `find | xargs grep` are all blocked.
GREP_COMMAND = re.compile(r"(?<![\w./-])(?:grep|rg|egrep|fgrep)\b")
BLOCKED_GREP = "grep and rg are not available as commands in this run, only as a filter after a pipe. Use the search tool to find text in files."
# a heredoc from its `<<WORD` line to the terminator line; the body is data (patch scripts, file contents), not commands
HEREDOC = re.compile(r"<<-?\s*(['\"]?)(\w+)\1[^\n]*\n.*?\n\s*\2(?=\s|$)", re.S)


def command_words(command):
    """The command with heredoc bodies cut out, so the block rules only see what bash runs as commands."""
    return HEREDOC.sub("<<HEREDOC", command)


def after_pipe(command, pos):
    """True if the command word at `pos` directly follows a single `|` (a filter, allowed by every block rule)."""
    before = command[:pos].rstrip()
    return before.endswith("|") and not before.endswith("||")


def runs_grep(command):
    """True if the bash command runs grep/rg as a command. A match right after a single `|` is a filter and allowed."""
    command = command_words(command)
    return any(not after_pipe(command, m.start()) for m in GREP_COMMAND.finditer(command))


# These five regexes are the parts of the rule (runs_read below) that decides whether a bash command reads a file.
# READ_COMMAND finds cat, head, nl or sed standing as a command, not inside a longer token like ./cat or my_cat
# (same word rule as GREP_COMMAND). SEGMENT_END marks where that command's arguments stop. WRITE_REDIRECT checks
# those arguments for a heredoc or a stdout redirect, because `cat > file` and `cat <<EOF` write a file. SED_QUIET
# checks for -n, because only `sed -n` prints lines and counts as reading. SED_IN_PLACE checks for -i, because
# `sed -i` edits the file, even with -n.
#
# tail is left out on purpose. In the bash-only GLM run it polls logs of background jobs in 1,667 of 1,818 calls,
# and the read tool has no way to read the end of a file (no negative offset). The pagers less and more are left
# out too, because nobody runs them in a non-interactive shell and "more" is an English word that shows up in
# commit messages.
READ_COMMAND = re.compile(r"(?<![\w./-])(?:cat|head|nl|sed)\b")
SEGMENT_END = re.compile(r"\|\|?|;|&&|\n")  # the next |, ;, && or line break
WRITE_REDIRECT = re.compile(r"<<|(?<!\d)>")  # 2> is stderr and does not count
SED_QUIET = re.compile(r"\s-(?:[a-zA-Z]*n|-quiet|-silent)")  # also bundled, like -En or -ne
SED_IN_PLACE = re.compile(r"\s-(?:[a-zA-Z]*i|-in-place)")
BLOCKED_READ = "cat, head, nl and sed -n are not available as commands in this run, only as a filter after a pipe. Use the read tool to read files."


def runs_read(command):
    """True if the bash command reads a file with cat, head, nl or sed -n as a command. Allowed: a filter after a
    single `|`, `cat` that writes (heredoc or redirect), and `sed` without -n or with -i (an edit, not a read)."""
    command = command_words(command)
    for m in READ_COMMAND.finditer(command):
        if after_pipe(command, m.start()):
            continue
        args = SEGMENT_END.split(command[m.end():], 1)[0]
        if m.group() == "cat" and WRITE_REDIRECT.search(args):
            continue
        if m.group() == "sed" and (not SED_QUIET.search(args) or SED_IN_PLACE.search(args)):
            continue
        return True
    return False


class ToolsModel(BashModel):
    """Extension. Adds the named tools from config.tools next to bash. A named tool call becomes a shell command
    that runs the tool's module in the container, so environment, observations and trajectories stay as in
    mini-swe-agent."""

    def __init__(self, **kwargs):
        super().__init__(config_class=ToolsModelConfig, **kwargs)

    def _tools(self):
        tools = [BASH_TOOL]
        for name in self.config.tools:
            tool = NAMED_TOOLS[name][0]
            if name == "search" and self.config.tool_directive:
                tool = copy.deepcopy(tool)
                tool["function"]["description"] = self.config.tool_directive.strip() + " " + tool["function"]["description"]
            tools.append(tool)
        return tools

    def _parse_actions(self, response):
        tool_calls = response.choices[0].message.tool_calls or []
        if not tool_calls:
            return super()._parse_actions(response)  # raises mini-swe-agent's "no tool calls" FormatError
        finish_reason = response.choices[0].finish_reason
        check_reasons(tool_calls, self.reason_fields, self.config.format_error_template, finish_reason)
        actions = []
        for tool_call in tool_calls:
            if tool_call.function.name in self.config.tools:
                # the arguments go to the tool's module as they are, it checks them and reports problems as the tool result
                module = NAMED_TOOLS[tool_call.function.name][1]
                command = shlex.join([sys.executable, "-m", module, tool_call.function.arguments])
                actions.append({"command": command, "tool_call_id": tool_call.id})
                continue
            # bash (and unknown names) get mini's own validation and error messages
            bash_actions = parse_toolcall_actions(
                [tool_call],
                format_error_template=self.config.format_error_template,
                template_kwargs={"finish_reason": finish_reason},
            )
            for action in bash_actions:
                if self.config.block_grep and runs_grep(action["command"]):
                    raise format_error(self.config.format_error_template, BLOCKED_GREP, finish_reason)
                if self.config.block_read and runs_read(action["command"]):
                    raise format_error(self.config.format_error_template, BLOCKED_READ, finish_reason)
            actions += bash_actions
        return actions

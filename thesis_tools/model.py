"""The model classes behind the conditions. Pier selects one with --ak model_class=thesis_tools.model.<class>. The
model class sends every LLM request and the tools list is part of it, so this is where the tool surface is set.

BashModel is the baseline, with bash as the only tool. ToolsModel adds the named tools from the `tools` yaml key
(search and/or read) next to bash. tool_directive is put in front of the search tool description, so it can change
in the yaml without a new package version.

reason_prompt adds a required `why_this_tool` string as the first parameter of every tool, and reason_against_prompt
one `why_not_<tool>` string per other tool. The tool names are in the field names so the model cannot mix up which
reason belongs to which tool. The fields are stored with the call in the trajectory and never reach the agent. A call
with a missing, blank or misnamed field is refused. After MAX_REASON_REFUSALS refused responses in a row such calls
run anyway, so a model that ignores the fields or gets them wrong does not loop and does not kill the trial.

block_grep and block_read (ToolsModel only) are the leave-one-out configurations. A bash command that runs grep or rg
(block_grep), or cat, head, nl or sed -n (block_read), is refused with a nudge to use the named tool instead. The
same commands as a filter after a pipe stay allowed (ps aux | grep, git log | head), and heredoc bodies are not
blocked.

When we refuse a call, we do not run it and return a message saying what to do instead, with exit code 127. We do
not return a format error, because mini-swe-agent ends a trial after 3 format errors in a row. The other calls in
the same response still run.

Both classes abort the trial on a context overflow.
"""

import copy
import json
import re
import shlex
import sys
import time

import litellm
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
    min_request_interval: float = 0
    """Seconds a trial waits at least between the starts of two model calls. Trials run in separate containers and
    cannot share a rate limiter, so each one paces itself: with n parallel trials and a quota of C requests per hour,
    n * 3600 / C keeps the whole run under the quota. 0 means no pacing (the default, fine without a quota)."""
    max_rate_limit_wait: float = 7200
    """Total seconds a trial may sleep on Retry-After waits (see rate_limit_wait) before the RateLimitError is
    raised and ends the trial."""


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


# after this many refused responses in a row, calls with missing reasons are no longer refused and just run, so
# the model cannot loop
MAX_REASON_REFUSALS = 3


def reason_gap(tool_call, fields):
    """Returns the refusal message if the call is missing a reason field, else None."""
    try:
        args = json.loads(tool_call.function.arguments)
    except Exception:
        return None
    if not isinstance(args, dict):
        return None
    name = tool_call.function.name
    missing = [f for f in fields.get(name, {}) if not str(args.get(f, "")).strip()]
    if not missing:
        return None
    text = f"Nothing in this {name} call was run. It is missing {', '.join(missing)}."
    wrong = [k for k in args if k.startswith("why_") and k not in fields[name]]
    if wrong:
        text += f" The {name} tool has no {', '.join(wrong)} argument."
    return text + " Send the call again with every why_ argument filled in."


RETRY_AFTER = re.compile(r"Retry after (\d+)s")


def rate_limit_wait(error, default=60):
    """Seconds to wait after a 429 from litellm's RateLimitError: the Retry-After header if present, else a
    "(Retry after Ns)" in the message, else `default`. Providers with hourly or daily quotas answer with waits of up to
    an hour; litellm's own backoff caps every sleep at 10 s and gives up long before that."""
    headers = getattr(error, "headers", None) or getattr(getattr(error, "response", None), "headers", None) or {}
    value = headers.get("retry-after") or headers.get("Retry-After")
    try:
        return float(value)
    except (TypeError, ValueError):
        pass
    match = RETRY_AFTER.search(str(error))
    return float(match.group(1)) if match else default


class BashModel(LitellmModel):
    """Baseline. A copy of mini-swe-agent's LitellmModel with two changes.

    1. tools=[BASH_TOOL] is not hardcoded, we send self.tools instead, so ToolsModel can add tools.
    2. the except block immediately stops the trial on a context overflow. Some providers send it as a plain
       BadRequestError, which mini retries until the rate limit or the timeout kills the trial.
       Re-raising it as ContextWindowExceededError makes mini abort right away.
    3. rate limits (optional, both default off): min_request_interval paces the calls so a run stays under a
       provider's request quota, and a RateLimitError that survives litellm's own retries (num_retries in
       model_kwargs) is slept out for the Retry-After time and retried instead of ending the trial. Not needed
       for a self-served model or a provider without quotas.
    """

    def __init__(self, *, config_class=BashModelConfig, **kwargs):
        super().__init__(config_class=config_class, **kwargs)
        self.last_request = -float("inf")
        tools = self._tools()
        names = [tool["function"]["name"] for tool in tools]
        self.reason_fields = reason_fields(names, self.config.reason_prompt, self.config.reason_against_prompt)
        self.tools = tuple(add_reasons(tool, self.reason_fields[name]) for tool, name in zip(tools, names))
        self.reason_refusals = 0

    def _tools(self):
        """The tool surface of this condition, before the reason parameter is added. Subclasses override."""
        return [BASH_TOOL]

    def _parse_actions(self, response):
        actions = super()._parse_actions(response)
        return self.refuse_reason_gaps(response.choices[0].message.tool_calls or [], actions)

    def refuse_reason_gaps(self, tool_calls, actions):
        """Refuses every call that is missing a reason field. A response with all fields filled in resets the
        count, and after MAX_REASON_REFUSALS refused responses in a row the calls run anyway."""
        gaps = {tool_call.id: reason_gap(tool_call, self.reason_fields) for tool_call in tool_calls}
        if not any(gaps.values()):
            self.reason_refusals = 0
            return actions
        if self.reason_refusals >= MAX_REASON_REFUSALS:
            return actions
        self.reason_refusals += 1
        for action in actions:
            if gaps.get(action["tool_call_id"]):
                action["command"] = refusal(gaps[action["tool_call_id"]])
        return actions

    def _query(self, messages, **kwargs):
        slept = 0
        while True:
            pause = self.last_request + self.config.min_request_interval - time.monotonic()
            if pause > 0:
                time.sleep(pause)
            self.last_request = time.monotonic()
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
            except litellm.exceptions.RateLimitError as e:
                wait = rate_limit_wait(e) + 5  # the window reopens at a clock boundary, 5 s of slack for clock skew
                if slept + wait > self.config.max_rate_limit_wait:
                    raise
                print(f"thesis_tools: rate limited, sleeping {wait:.0f}s ({str(e)[:200]})", file=sys.stderr, flush=True)
                time.sleep(wait)
                slept += wait


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


def refusal(message):
    """We run this instead of a blocked bash command or a call with a reason gap. It just prints the text telling the
    agent what to do instead and exits with 127, so the agent sees a normal failed command and not a format error."""
    return f"printf '%s\\n' {shlex.quote(message)}; exit 127"


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
        actions = []
        for tool_call in tool_calls:
            if tool_call.function.name in self.config.tools:
                # the module checks the arguments itself and reports problems as its output
                module = NAMED_TOOLS[tool_call.function.name][1]
                command = shlex.join([sys.executable, "-m", module, tool_call.function.arguments])
                actions.append({"command": command, "tool_call_id": tool_call.id})
                continue
            # bash and unknown tool names get mini-swe-agent's own checks
            bash_actions = parse_toolcall_actions(
                [tool_call],
                format_error_template=self.config.format_error_template,
                template_kwargs={"finish_reason": finish_reason},
            )
            for action in bash_actions:
                if self.config.block_grep and runs_grep(action["command"]):
                    action["command"] = refusal(BLOCKED_GREP)
                elif self.config.block_read and runs_read(action["command"]):
                    action["command"] = refusal(BLOCKED_READ)
            actions += bash_actions
        return self.refuse_reason_gaps(tool_calls, actions)

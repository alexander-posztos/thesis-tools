# thesis-tools

Named tools for [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) 2.4.6, built for a master thesis on how an LLM coding agent's tool surface affects its SWE-bench-style performance (TU Wien, 2026).

## Layout

```
thesis_tools/search.py   the search tool and its CLI (python -m thesis_tools.search '<json args>')
thesis_tools/read.py     the read tool and its CLI (python -m thesis_tools.read '<json args>')
thesis_tools/args.py     argument checking shared by the tool CLIs
thesis_tools/model.py    BashModel and ToolsModel
bash.yaml, tools.yaml    mini-swe-agent config, default prompt plus a one-line search-tool mention
minimal*.yaml            shared minimal prompt, the directive variant steers via the tool description,
                         the reason variants ask for a stated reason in every tool call, the nogrep and
                         nocat variants block the bash path of the named tool
tests/                   pytest
```

## Usage

The package is installed into the task image alongside mini-swe-agent, and the tool surface is selected through mini-swe-agent's `model_class` and config file. With Pier (DeepSWE runner):

```
--ak 'extra_python_packages=["git+https://github.com/alexander-posztos/thesis-tools.git"]'
--ak config_file=tools.yaml --ak model_class=thesis_tools.model.ToolsModel
```

Use `bash.yaml` and `BashModel` for the baseline.

`minimal.yaml` replaces mini-swe-agent's default template with a shorter one that doesn't name tools. It works with both model classes.

`minimal-directive.yaml` is the same file plus `tool_directive`, a sentence that is put in front of the search tool description.

`minimal-reason.yaml` is the same file plus `reason_prompt` (and `reason_prompt_single` for a surface with one tool) and one paragraph in the instance template. When set, every tool (bash in `BashModel`, bash and search in `ToolsModel`) gets a required `why_this_tool` string as its first parameter, described by that sentence. The model states why it picked the tool, and why not the others, before it writes the arguments. A call that leaves the field out or blank gets a format error and the model retries. The field stays in the trajectory with the tool call arguments and is never passed to the container.

`minimal-reason-nogrep.yaml` is `minimal-reason.yaml` plus `block_grep` (ToolsModel only) and one sentence in the instance template. Bash commands that run grep or rg as a command get a format error that points to the search tool, so the search tool is the only way to search file contents; grep as a filter after a pipe (`ps aux | grep x`) stays allowed, and heredoc bodies are not inspected. This is the leave-one-out arm.

`minimal-reason-read.yaml` is `minimal-reason.yaml` with `tools: [read]`, so the surface is bash plus the read tool instead of bash plus search. `tools` lists the named tools next to bash (`search`, `read`, or both, in that order); it defaults to `[search]`, so the older configs are unchanged.

`minimal-search-read.yaml` is `minimal.yaml` with `tools: [search, read]`, the free-choice arm with both named tools and nothing else (no reason fields, directive or blocking).

Two optional keys on both model classes deal with provider rate limits, for example an hourly request quota that answers 429 with a Retry-After of up to an hour. `min_request_interval` is the minimum number of seconds between the starts of two model calls in a trial; with n parallel trials and a quota of C requests per hour, n * 3600 / C keeps the run under the quota. A 429 that survives litellm's own retries (`num_retries` in `model_kwargs`, backoff capped at 10 s) is slept out for the Retry-After time (header, else a "Retry after Ns" in the message) and retried, up to `max_rate_limit_wait` seconds per trial (default 7200), instead of ending the trial. Both are off by default and not needed for a self-served model or a provider without quotas.

`minimal-reason-read-nocat.yaml` is `minimal-reason-read.yaml` plus `block_read` and one sentence in the instance template. Bash commands that run cat, head, nl or sed -n as a command get a format error that points to the read tool. Allowed: the same programs as a filter after a pipe, `cat` that writes (heredoc or redirect), `sed` without `-n` or with `-i`, and `tail` (which polls logs in the bash-only trajectories and has no read-tool equivalent).

## Tools

Both tools run in the task container as a shell command built from the tool call's JSON arguments, so observations and trajectories stay as in mini-swe-agent. Bad arguments are reported as the tool result in pi's message format.

`search` is a port of pi's grep tool on ripgrep: pattern, optional path, glob, ignore_case, literal, context, limit. Output is `path:line: text`, capped at the match limit, 50KB and 500 characters per line.

`read` is a port of pi's read tool for text files: path, optional 1-indexed offset and limit. Every output line is prefixed with its line number (`12: text`, unlike pi). Output is capped at 2000 lines or 50KB, whichever comes first, never cutting a line, with a notice that names the next offset.


# thesis-tools

Named tools for [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) 2.4.6, built for a master thesis on how an LLM coding agent's tool surface affects its SWE-bench-style performance (TU Wien, 2026).

## Layout

```
thesis_tools/search.py   the search tool and its CLI (python -m thesis_tools.search '<json args>')
thesis_tools/model.py    BashModel and ToolsModel
bash.yaml, tools.yaml    mini-swe-agent config, default prompt plus a one-line search-tool mention
minimal*.yaml            shared minimal prompt, the directive variant steers via the tool description,
                         the reason variant asks for a stated reason in every tool call
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


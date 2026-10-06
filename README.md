# thesis-tools

Named tools for [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) 2.4.6, built for a master thesis on how an LLM coding agent's tool surface affects its SWE-bench-style performance (TU Wien, 2026).

## Layout

```
thesis_tools/search.py   the search tool and its CLI (python -m thesis_tools.search '<json args>')
thesis_tools/model.py    BashModel and ToolsModel
bash.yaml, tools.yaml    mini-swe-agent config 
tests/                   pytest
```

## Usage

The package is installed into the task image along mini-swe-agent, and the tool surface is selected through mini-swe-agent's `model_class` and config file. With Pier (DeepSWE runner):

```
--ak extra_python_packages=git+https://github.com/alexander-posztos/thesis-tools.git
--ak config_file=tools.yaml --ak model_class=thesis_tools.model.ToolsModel
```

Use `bash.yaml` and `BashModel` for the baseline.


"""Search tool. A Python port of pi's grep tool (earendil-works/pi @ 98d2e19, MIT,
Copyright (c) 2025 Mario Zechner). ToolsModel runs it in the task container as
`python -m thesis_tools.search '<json args>'`. Differences from pi are marked "Deviation".
"""

import functools
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

DEFAULT_LIMIT = 100
DEFAULT_MAX_BYTES = 50 * 1024
MAX_LINE_LENGTH = 500

RG = Path(sys.executable).parent / "rg"

SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search",
        "description": (
            "Search file contents for a pattern. Returns matching lines with file paths and line numbers. "
            f"Respects .gitignore. Output is truncated to {DEFAULT_LIMIT} matches or "
            f"{DEFAULT_MAX_BYTES // 1024}KB (whichever is hit first). "
            f"Long lines are truncated to {MAX_LINE_LENGTH} chars."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Search pattern (regex or literal string)"},
                "path": {
                    "type": "string",
                    "description": "Directory or file to search (default: current directory)",
                },
                "glob": {
                    "type": "string",
                    "description": "Filter files by glob pattern, e.g. '*.ts' or '**/*.spec.ts'",
                },
                "ignoreCase": {"type": "boolean", "description": "Case-insensitive search (default: false)"},
                "literal": {
                    "type": "boolean",
                    "description": "Treat pattern as literal string instead of regex (default: false)",
                },
                "context": {
                    "type": "number",
                    "description": "Number of lines to show before and after each match (default: 0)",
                },
                "limit": {"type": "number", "description": "Maximum number of matches to return (default: 100)"},
            },
            "required": ["pattern"],
        },
    },
}

class SearchError(Exception):
    """An error the model sees as the tool result. Any other exception is a bug and crashes."""


@functools.cache
def file_lines(file_path):
    """Reads a file's lines, cached so a file with many matches is read only once. Empty if unreadable."""
    try:
        # newline="" so only "\n" ends a line, like in rg. otherwise a lone "\r" shifts the line numbers.
        # Deviation: pi splits on "\r" too, which misaligns context lines in old Mac-style files.
        with open(file_path, encoding="utf-8", errors="replace", newline="") as f:
            return f.read().split("\n")
    except OSError:
        return []


def truncate_line(line):
    if len(line) <= MAX_LINE_LENGTH:
        return line, False
    return f"{line[:MAX_LINE_LENGTH]}... [truncated]", True


def truncate_head(content):
    """Cuts the whole output at 50KB, never mid-line. Returns (text, whether it cut)."""
    if len(content.encode()) <= DEFAULT_MAX_BYTES:
        return content, False
    kept, size = [], 0
    for i, line in enumerate(content.split("\n")):
        size += len(line.encode()) + (1 if i else 0)
        if size > DEFAULT_MAX_BYTES:
            break
        kept.append(line)
    return "\n".join(kept), True


def search(pattern, path=None, glob=None, ignoreCase=False, literal=False, context=0, limit=DEFAULT_LIMIT):
    """Runs rg and returns the text the model sees. Expected problems (bad path, bad regex) raise SearchError."""
    search_path = os.path.abspath(os.path.expanduser(path or "."))
    if not os.path.exists(search_path):
        raise SearchError(f"Path not found: {search_path}")
    is_dir = os.path.isdir(search_path)

    # how many lines to show above and below each match
    context = max(int(context), 0)

    # maximum number of matches to return
    limit = max(int(limit), 1)

    args = [RG, "--json", "--line-number", "--color=never", "--hidden"]
    if ignoreCase:
        args.append("--ignore-case")
    if literal:
        args.append("--fixed-strings")
    if glob:
        args += ["--glob", glob]
    # Deviation: we exclude .git since hits in there aren't useful. pi doesn't, but most
    # other harnesses do it this way. Has to come after the user's glob, the last glob wins in rg.
    args += ["--glob", "!.git"]
    args += ["--", pattern, search_path]

    # we use the temporary file to write error messages into, since we only read stdout. a stderr pipe would fill up at 64KB
    with tempfile.TemporaryFile() as stderr:
        proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=stderr)
        matches, limit_reached = [], False
        for line in proc.stdout:
            event = json.loads(line)
            if event["type"] != "match":
                continue
            data = event["data"]
            # rg sends "bytes" instead of "text" for paths that aren't valid UTF-8
            file_path = data["path"].get("text")
            if file_path:
                matches.append((file_path, data["line_number"]))
            if len(matches) >= limit:
                limit_reached = True
                proc.kill()
                break
        proc.stdout.close()
        code = proc.wait()
        stderr.seek(0)
        stderr_text = stderr.read().decode(errors="replace").strip()

    if not limit_reached and code not in (0, 1):
        raise SearchError(stderr_text or f"ripgrep exited with code {code}")
    if not matches:
        return "No matches found"

    lines_truncated = False
    output_lines = []
    # search is done, now we format each match as one line like "src/app.py:12: def target():".
    # the model only sees this text, so it needs the file and line number to find the match again.
    for file_path, line_number in matches:
        # shorten the full path rg gives us. for a folder search, the path from that folder. for a file search, just the file name.
        relative = os.path.relpath(file_path, search_path) if is_dir else os.path.basename(file_path)
        lines = file_lines(file_path)
        if not lines:
            output_lines.append(f"{relative}:{line_number}: (unable to read file)")
            continue
        # print the match plus `context` lines above and below. ":" marks the match, "-" the lines around it
        for current in range(max(1, line_number - context), min(len(lines), line_number + context) + 1):
            text, cut = truncate_line(lines[current - 1].replace("\r", ""))
            lines_truncated |= cut
            sep = ":" if current == line_number else "-"
            output_lines.append(f"{relative}{sep}{current}{sep} {text}")

    output, bytes_truncated = truncate_head("\n".join(output_lines))
    notices = []
    if limit_reached:
        notices.append(f"{limit} matches limit reached. Use limit={limit * 2} for more, or refine pattern")
    if bytes_truncated:
        notices.append(f"{DEFAULT_MAX_BYTES / 1024:.1f}KB limit reached")
    if lines_truncated:
        # Deviation: pi says "Use read tool to see full lines", and we dont have a read tool yet
        notices.append(f"Some lines truncated to {MAX_LINE_LENGTH} chars. Use bash to see full lines")
    if notices:
        output += f"\n\n[{'. '.join(notices)}]"
    return output


def validate(raw_args):
    """Parses the JSON arguments, checks them against SEARCH_TOOL's schema and returns the kwargs for
    search(). Bad arguments become the tool result, same message format as pi. No type conversion."""
    try:
        args = json.loads(raw_args)
    except ValueError as e:
        raise SearchError(f"Error parsing tool call arguments: {e}.") from e
    if not isinstance(args, dict):
        raise SearchError("Tool call arguments must be a JSON object.")
    props = SEARCH_TOOL["function"]["parameters"]["properties"]
    types = {"string": str, "boolean": bool, "number": (int, float)}
    errors = [] if "pattern" in args else ["  - /pattern: Expected required property"]
    for key, value in args.items():
        if key in props and not isinstance(value, types[props[key]["type"]]):
            errors.append(f"  - /{key}: Expected {props[key]['type']}")
    if errors:
        raise SearchError(
            'Validation failed for tool "search":\n' + "\n".join(errors)
            + f"\n\nReceived arguments:\n{json.dumps(args, indent=2)}"
        )
    # unknown keys are ignored, like in pi
    return {key: value for key, value in args.items() if key in props}


def main():
    # always write UTF-8, so a container with an odd locale can't crash us on non-ASCII matches
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    try:
        sys.stdout.write(search(**validate(sys.argv[1])))
    except SearchError as e:
        sys.stdout.write(str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()

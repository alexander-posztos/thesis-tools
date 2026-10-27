"""Read tool. A Python port of pi's read tool (earendil-works/pi @ 98d2e19 ),
text files only. ToolsModel runs it in the task container as `python -m thesis_tools.read '<json args>'`.
Differences from pi are marked "Deviation". Left out on purpose: images, the SSH-style pluggable file operations,
abort signals, the UI renderers and pi's macOS filename variants (NFD, curly quotes, the no-break space in
screenshot names), none of which exist in a benchmark container.
"""

import os
import sys

from thesis_tools.args import validate as validate_args

DEFAULT_MAX_LINES = 2000
DEFAULT_MAX_BYTES = 50 * 1024

READ_TOOL = {
    "type": "function",
    "function": {
        "name": "read",
        "description": (
            "Read the contents of a file. Each line is prefixed with its line number. "
            f"Output is truncated to {DEFAULT_MAX_LINES} lines or {DEFAULT_MAX_BYTES // 1024}KB (whichever is hit "
            "first). Use offset/limit for large files. When you need the full file, continue with offset until complete."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file to read (relative or absolute)"},
                "offset": {"type": "number", "description": "Line number to start reading from (1-indexed)"},
                "limit": {"type": "number", "description": "Maximum number of lines to read"},
            },
            "required": ["path"],
        },
    },
}


class ReadError(Exception):
    """An error the model sees as the tool result. Any other exception is a bug and crashes."""


def format_size(size):
    if size < 1024:
        return f"{size}B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f}KB"
    return f"{size / (1024 * 1024):.1f}MB"


def truncate_head(lines):
    """Keeps the first lines that fit into DEFAULT_MAX_LINES and DEFAULT_MAX_BYTES, never a partial line. Returns
    (kept lines, what cut them: None, "lines", "bytes" or "first line" when even the first line is over the byte cap)."""
    if len(lines) <= DEFAULT_MAX_LINES and sum(len(line.encode()) + 1 for line in lines) - 1 <= DEFAULT_MAX_BYTES:
        return lines, None
    if len(lines[0].encode()) > DEFAULT_MAX_BYTES:
        return [], "first line"
    kept, size = [], 0
    for line in lines[:DEFAULT_MAX_LINES]:
        size += len(line.encode()) + (1 if kept else 0)
        if size > DEFAULT_MAX_BYTES:
            return kept, "bytes"
        kept.append(line)
    return kept, "lines"


def read(path, offset=None, limit=None):
    """Returns the text the model sees. Expected problems (missing file, offset past the end) raise ReadError."""
    file_path = os.path.abspath(os.path.expanduser(path))
    if os.path.isdir(file_path):
        raise ReadError(f"Path is a directory, not a file: {path}")
    try:
        with open(file_path, encoding="utf-8", errors="replace", newline="") as f:
            text = f.read()
    except OSError as e:
        raise ReadError(f"Cannot read {path}: {e.strerror}") from e

    # Deviation: pi counts the empty string after a final newline as a line, so its "of N" is one too high
    # for every file that ends in a newline. We count lines the way nl and wc -l do.
    if not text:
        return "(empty file)"  # Deviation: pi returns the empty string, which models read as a failed call
    all_lines = text.split("\n")
    if text.endswith("\n"):
        all_lines.pop()

    # 1-indexed offset to a 0-indexed start
    start = max(int(offset) - 1, 0) if offset else 0
    if start >= len(all_lines):
        raise ReadError(f"Offset {offset} is beyond end of file ({len(all_lines)} lines total)")
    end = min(start + int(limit), len(all_lines)) if limit is not None else len(all_lines)

    # Deviation: pi returns the raw text. We prefix every line with its number, as OpenCode and Claude Code do and as
    # cat -n would, because the search tool reports file:line and this surface has no edit tool that needs the
    # exact raw text. The caps apply to the numbered output, which is what the model gets.
    numbered = [f"{n}: {line}" for n, line in enumerate(all_lines[start:end], start + 1)]
    kept, cut_by = truncate_head(numbered)
    if cut_by == "first line":
        size = format_size(len(all_lines[start].encode()))
        return (
            f"[Line {start + 1} is {size}, exceeds {format_size(DEFAULT_MAX_BYTES)} limit. "
            f"Use bash: sed -n '{start + 1}p' {path} | head -c {DEFAULT_MAX_BYTES}]"
        )
    output = "\n".join(kept)
    last = start + len(kept)
    if cut_by:
        cap = f" ({format_size(DEFAULT_MAX_BYTES)} limit)" if cut_by == "bytes" else ""
        output += f"\n\n[Showing lines {start + 1}-{last} of {len(all_lines)}{cap}. Use offset={last + 1} to continue.]"
    elif last < len(all_lines):
        output += f"\n\n[{len(all_lines) - last} more lines in file. Use offset={last + 1} to continue.]"
    return output


def validate(raw_args):
    """Checks the JSON arguments against READ_TOOL's schema, see args.validate."""
    return validate_args(raw_args, READ_TOOL, ReadError)


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    try:
        sys.stdout.write(read(**validate(sys.argv[1])))
    except ReadError as e:
        sys.stdout.write(str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()

import pytest

from thesis_tools.search import SearchError, search, validate


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_match_set(tmp_path, monkeypatch):
    # rg only honors .gitignore inside a git repo, and a bare .git/ dir is enough for that
    write(tmp_path / ".git" / "COMMIT_EDITMSG", "target\n")
    write(tmp_path / ".gitignore", "build/\n")
    write(tmp_path / "build" / "out.py", "target\n")
    write(tmp_path / ".hidden" / "conf.py", "target = 1\n")
    write(tmp_path / "src" / "app.py", "def target():\n    return 1\n")
    monkeypatch.chdir(tmp_path)

    # hidden files are searched, gitignored files and .git are not. order across files isn't fixed
    assert set(search("target").split("\n")) == {".hidden/conf.py:1: target = 1", "src/app.py:1: def target():"}
    # a file path prints as its basename, context lines use "-" instead of ":"
    assert search("target", path="src/app.py", context=1) == "app.py:1: def target():\napp.py-2-     return 1"
    assert search("nothing_here") == "No matches found"


def test_line_endings(tmp_path):
    # a lone "\r" is not a line break for rg, so it must not shift our line numbers either
    write(tmp_path / "cr.txt", "x\rhit\nnext\n")
    assert search("hit", path=str(tmp_path / "cr.txt"), context=1) == "cr.txt:1: xhit\ncr.txt-2- next"


def test_caps(tmp_path):
    write(tmp_path / "many.txt", "hit\n" * 150)
    write(tmp_path / "long.txt", "hit" + "x" * 600 + "\n")

    out = search("hit", path=str(tmp_path / "many.txt"))
    assert out == "\n".join(f"many.txt:{i}: hit" for i in range(1, 101)) + (
        "\n\n[100 matches limit reached. Use limit=200 for more, or refine pattern]"
    )
    out = search("hit", path=str(tmp_path / "long.txt"))
    assert out == "long.txt:1: hit" + "x" * 497 + "... [truncated]" + (
        "\n\n[Some lines truncated to 500 chars. Use bash to see full lines]"
    )


def test_validate():
    assert validate('{"pattern": "x", "extra": 1}') == {"pattern": "x"}
    for raw, message in [
        ('{"limit": 5}', "Expected required property"),
        ('{"pattern": "x", "limit": "5"}', "Expected number"),
        ('{"pattern": ', "Error parsing tool call arguments"),
        ("[1, 2]", "must be a JSON object"),
    ]:
        with pytest.raises(SearchError, match=message):
            validate(raw)

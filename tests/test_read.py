import pytest

from thesis_tools.read import DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES, ReadError, read, validate


def test_read(tmp_path):
    f = tmp_path / "a.py"
    f.write_text("one\ntwo\nthree\n")
    # the trailing newline does not count as a line (pi counts it), so "of N" matches wc -l
    assert read(str(f)) == "1: one\n2: two\n3: three"
    assert read(str(f), offset=2) == "2: two\n3: three"
    assert read(str(f), offset=2, limit=1) == "2: two\n\n[1 more lines in file. Use offset=3 to continue.]"
    assert read(str(f), offset=0, limit=10) == "1: one\n2: two\n3: three"  # offset is 1-indexed, 0 reads from the start
    (tmp_path / "empty").write_text("")
    assert read(str(tmp_path / "empty")) == "(empty file)"
    (tmp_path / "crlf").write_bytes(b"a\r\nb")
    assert read(str(tmp_path / "crlf")) == "1: a\r\n2: b"  # only \n splits lines, like pi


def test_errors(tmp_path):
    (tmp_path / "a").write_text("x\n")
    with pytest.raises(ReadError, match=r"Offset 5 is beyond end of file \(1 lines total\)"):
        read(str(tmp_path / "a"), offset=5)
    with pytest.raises(ReadError, match="No such file"):
        read(str(tmp_path / "missing"))
    with pytest.raises(ReadError, match="is a directory"):
        read(str(tmp_path))


def test_caps(tmp_path):
    many = tmp_path / "many"
    many.write_text("x\n" * (DEFAULT_MAX_LINES + 5))
    out = read(str(many))
    assert out.endswith(f"\n{DEFAULT_MAX_LINES}: x\n\n[Showing lines 1-{DEFAULT_MAX_LINES} of {DEFAULT_MAX_LINES + 5}. "
                        f"Use offset={DEFAULT_MAX_LINES + 1} to continue.]")
    assert read(str(many), offset=DEFAULT_MAX_LINES + 1) == "\n".join(f"{i}: x" for i in range(2001, 2006))

    # the byte cap counts the numbered output and never cuts a line in half
    big = tmp_path / "big"
    big.write_text(("y" * 1000 + "\n") * 60)
    out = read(str(big))
    body, notice = out.split("\n\n")
    assert len(body.encode()) <= DEFAULT_MAX_BYTES and body.count("\n") == 49  # 50 lines fit, 51 would not
    assert notice == "[Showing lines 1-50 of 60 (50.0KB limit). Use offset=51 to continue.]"

    huge = tmp_path / "huge"
    huge.write_text("a\n" + "z" * (DEFAULT_MAX_BYTES + 1) + "\n")
    assert read(str(huge), offset=2) == (
        f"[Line 2 is 50.0KB, exceeds 50.0KB limit. Use bash: sed -n '2p' {huge} | head -c {DEFAULT_MAX_BYTES}]"
    )


def test_validate():
    assert validate('{"path": "x", "why_this_tool": "because"}') == {"path": "x"}
    for raw, message in [
        ('{"offset": 5}', "Expected required property"),
        ('{"path": "x", "limit": "5"}', "Expected number"),
        ('{"path": ', "Error parsing tool call arguments"),
    ]:
        with pytest.raises(ReadError, match=message):
            validate(raw)

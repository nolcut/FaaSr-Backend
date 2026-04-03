# src/faasr_ai/vim_editor.py
from __future__ import annotations

import json
import os
import subprocess
import tempfile


def open_vim_for_input(header: str = "", initial_content: str = "") -> str:
    """Open vim with a .md tempfile for the user to write text.

    Lines starting with ``#`` are stripped from the returned result
    so they can be used as instructional comments in the header.
    """
    with tempfile.NamedTemporaryFile(
        suffix=".txt", mode="w", delete=False
    ) as f:
        f.write(header + initial_content)
        tmp_path = f.name

    subprocess.call(["vim", tmp_path])

    with open(tmp_path, "r") as f:
        content = f.read()

    os.unlink(tmp_path)

    lines = [line for line in content.splitlines() if not line.strip().startswith("#")]
    return "\n".join(lines).strip()


def open_vim_for_json(json_data: dict, header_comment: str = "") -> dict:
    """Open vim with a clean .json tempfile for the user to review/edit.

    Instructions are printed to the terminal (not written into the file)
    so the JSON file is valid and renders cleanly in vim.
    Returns the parsed (possibly edited) JSON dict.
    """
    if header_comment:
        print(f"\n{header_comment}")

    content = json.dumps(json_data, indent=2) + "\n"

    with tempfile.NamedTemporaryFile(
        suffix=".json", mode="w", delete=False
    ) as f:
        f.write(content)
        tmp_path = f.name

    subprocess.call(["vim", tmp_path])

    with open(tmp_path, "r") as f:
        raw = f.read()

    os.unlink(tmp_path)

    return json.loads(raw)

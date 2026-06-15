#!/usr/bin/env python3
#
# Copyright 2026 René Widera
# SPDX-License-Identifier: MPL-2.0

from pathlib import Path
import re
import sys


NON_ASCII = re.compile(r"[^\x00-\x7F]")

def find_header_end(text: str) -> int:
    """Return the character offset after the initial license/author block."""

    if text.startswith("/*"):
        end = text.find("*/")
        if end != -1:
            return end + 2

    elif text.startswith("//"):
        offset = 0
        for line in text.splitlines(keepends=True):
            if line.startswith("//"):
                offset += len(line)
            else:
                break
        return offset

    return 0


def check_file(filename: str) -> bool:
    try:
        text = Path(filename).read_text(encoding="utf-8")
    except UnicodeDecodeError as e:
        print(f"{filename}: invalid UTF-8: {e}")
        return False

    allowed_end = find_header_end(text)

    header = text[:allowed_end]
    remainder = text[allowed_end:]

    start_line = header.count("\n") + 1
    found_non_ascii = False

    for line_offset, line in enumerate(remainder.splitlines(), start_line):
        for match in NON_ASCII.finditer(line):
            ch = match.group(0)
            col_no = match.start() + 1

            print(
                f"{filename}:{line_offset}:{col_no}: "
                f"unsupported non-ASCII character U+{ord(ch):04X} ({ch!r})"
            )
            found_non_ascii = True

    return not found_non_ascii


def main() -> int:
    if len(sys.argv) < 2:
        print(f"usage: {sys.argv[0]} <files...>", file=sys.stderr)
        return 1

    all_ok = True

    for filename in sys.argv[1:]:
        if not check_file(filename):
            all_ok = False

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())

"""Generate the cheatsheet PDF used by the documentation download link."""

from __future__ import annotations

import os
import pathlib
import re
import shutil
import subprocess
import textwrap

from sphinx.util import logging

logger = logging.getLogger(__name__)


def should_generate(app) -> bool:
    """Return whether cheatsheet.pdf must be generated."""
    value = os.getenv("ALPAKA_DOC_CHEATSHEET")
    if value is not None:
        if value in {"1", "ON"}:
            logger.info("Cheatsheet: force build via ALPAKA_DOC_CHEATSHEET=%s", value)
            return True
        if value in {"0", "OFF"}:
            logger.info("Cheatsheet: build disabled via ALPAKA_DOC_CHEATSHEET=%s", value)
            return False
        raise RuntimeError(f"Invalid ALPAKA_DOC_CHEATSHEET={value!r}; expected 0, 1, OFF, or ON")

    if shutil.which("rst2pdf") is None:
        logger.warning("Cheatsheet: 'rst2pdf' not found; skipping PDF generation.")
        return False

    source = pathlib.Path(app.srcdir) / "basic" / "cheatsheet.rst"
    snippet = pathlib.Path(app.srcdir).parent / "snippets" / "cheatsheet" / "cheatsheet.cpp"
    generator = pathlib.Path(__file__).resolve()
    style = pathlib.Path(app.confdir) / "basic" / "cheatsheet.style"
    output = pathlib.Path(app.builder.outdir) / "cheatsheet.pdf"

    if not output.exists():
        logger.info("Cheatsheet: build because cheatsheet.pdf does not exist.")
        return True

    return any(path.stat().st_mtime > output.stat().st_mtime for path in (source, snippet, generator, style))


def _extract_code(
    path: pathlib.Path,
    start_after: str | None,
    end_before: str | None,
    dedent: str | None,
) -> str:
    """Read and prepare the contents of a literalinclude directive."""
    lines = path.read_text(encoding="utf-8").splitlines()

    if start_after:
        begin = next(index for index, line in enumerate(lines) if start_after in line)
        lines = lines[begin + 1 :]

    if end_before:
        end = next(index for index, line in enumerate(lines) if end_before in line)
        lines = lines[:end]

    if dedent == "":
        # ``:dedent:`` without a value removes common indentation.
        lines = textwrap.dedent("\n".join(lines)).splitlines()
    elif dedent is not None:
        # ``:dedent: N`` removes up to N leading whitespace characters.
        amount = int(dedent)
        lines = [re.sub(rf"^[ \t]{{0,{amount}}}", "", line) for line in lines]

    return "\n".join(f"   {line}" if line else "" for line in lines)


def expand_source(source: pathlib.Path) -> str:
    """Convert Sphinx-specific markup into input suitable for rst2pdf."""
    lines = source.read_text(encoding="utf-8").splitlines()
    result: list[str] = []
    index = 0

    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        indent = line[: len(line) - len(line.lstrip())]

        if stripped.startswith(".. literalinclude::"):
            include = stripped.removeprefix(".. literalinclude::").strip()
            options: dict[str, str] = {}
            index += 1

            while index < len(lines) and lines[index].strip().startswith(":"):
                name, separator, value = lines[index].strip()[1:].partition(":")
                if not separator:
                    raise RuntimeError(f"Invalid literalinclude option: {lines[index]!r}")
                options[name] = value.strip()
                index += 1

            code = _extract_code(
                (source.parent / include).resolve(),
                options.get("start-after"),
                options.get("end-before"),
                options.get("dedent"),
            )

            result.extend(
                (
                    f"{indent}.. code-block:: c++",
                    "",
                    code,
                )
            )
            continue

        if stripped.startswith(".. only::"):
            block_indent = len(line) - len(line.lstrip())
            index += 1

            while index < len(lines):
                current = lines[index]

                if not current.strip():
                    index += 1
                    continue

                current_indent = len(current) - len(current.lstrip())
                if current_indent <= block_indent:
                    break

                index += 1

            continue

        # Remove inline Sphinx download roles.
        line = re.sub(r":download:`[^`]*`", "", line)

        result.append(line)
        index += 1

    return "\n".join(result) + "\n"


def generate_cheatsheet(app) -> None:
    """Generate cheatsheet.pdf when necessary."""
    if not should_generate(app):
        return

    rst2pdf = shutil.which("rst2pdf")
    if rst2pdf is None:
        return

    source = pathlib.Path(app.srcdir) / "basic" / "cheatsheet.rst"
    output = pathlib.Path(app.builder.outdir) / "cheatsheet.pdf"
    staged = pathlib.Path(app.builder.doctreedir).parent / "cheatsheet.rst"
    style = pathlib.Path(app.confdir) / "basic" / "cheatsheet.style"

    staged.parent.mkdir(parents=True, exist_ok=True)
    staged.write_text(expand_source(source), encoding="utf-8")

    logger.info("Cheatsheet: generating %s", output)

    generation = subprocess.run(
        [
            rst2pdf,
            "-s",
            str(style),
            str(staged),
            "-o",
            str(output),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if generation.returncode != 0:
        raise RuntimeError(f"Cheatsheet: rst2pdf failed with exit code {generation.returncode}:\n{generation.stderr}")

    if not output.exists():
        raise RuntimeError("Cheatsheet: rst2pdf did not create cheatsheet.pdf")

    logger.info("Cheatsheet: generated %s", output)

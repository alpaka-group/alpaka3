from __future__ import annotations

from pathlib import Path
import re

from docutils import nodes
from docutils.parsers.rst import Directive, directives


MARKER_RE = re.compile(r"^\s*//\s*(BEGIN|END)-[A-Za-z0-9_-]+\s*$")


class FilteredLiteralInclude(Directive):
    required_arguments = 1
    optional_arguments = 0
    final_argument_whitespace = False
    has_content = False
    option_spec = {
        "language": directives.unchanged_required,
        "linenos": directives.flag,
    }

    def run(self):
        env = self.state.document.settings.env
        _, filename = env.relfn2path(self.arguments[0])
        env.note_dependency(filename)

        path = Path(filename)
        lines = path.read_text(encoding="utf-8").splitlines()
        filtered = [line for line in lines if not MARKER_RE.match(line)]
        text = "\n".join(filtered)

        literal = nodes.literal_block(text, text, source=str(path))
        literal["language"] = self.options.get("language", "text")
        literal["linenos"] = "linenos" in self.options
        return [literal]


# Include a file into a rst file without the labels `[BEGIN|END]_*`
#
# Is used similar to `literalinclude`
def setup(app):
    app.add_directive("filteredliteralinclude", FilteredLiteralInclude)
    return {
        "version": "1.0",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }

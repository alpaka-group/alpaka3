"""Generate the math function reference block used by the tutorials."""

from __future__ import annotations

import pathlib
import xml.etree.ElementTree as ET

DOXYGEN_NAMESPACE_PAGE = "../doxygen/namespacealpaka_1_1math.html"


def _load_function_data(xml_file_path: pathlib.Path) -> dict[str, tuple[str, int]]:
    root = ET.parse(xml_file_path).getroot()

    function_data: dict[str, tuple[str, int]] = {}
    for member in root.findall(".//memberdef[@kind='function']"):
        function_name = member.findtext("name")
        member_id = member.get("id")
        if function_name and member_id:
            anchor = member_id.rsplit("_1", maxsplit=1)[-1]
            function_data[function_name] = (f"{DOXYGEN_NAMESPACE_PAGE}#{anchor}", len(member.findall("param")))

    return function_data


def _group_functions(
        function_data: dict[str, tuple[str, int]]
) -> dict[str, list[str]]:
    grouped_functions: dict[str, list[str]] = {
        "Unary Functions": [],
        "Binary Functions": [],
        "Other Functions": [],
    }

    for function_name, (_, param_count) in sorted(function_data.items()):
        if param_count == 1:
            grouped_functions["Unary Functions"].append(function_name)
        elif param_count == 2:
            grouped_functions["Binary Functions"].append(function_name)
        else:
            grouped_functions["Other Functions"].append(function_name)

    return grouped_functions


def generate_math_function_families(app) -> None:
    """Create the generated RST snippet referenced by the math tutorial."""

    output_path = pathlib.Path(app.srcdir, "_generated", "math_function_families.rst")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    xml_file_path = pathlib.Path(app.confdir).parent / pathlib.Path("doxygen", "xml", "namespacealpaka_1_1math.xml")

    function_data = _load_function_data(xml_file_path)
    grouped_functions = _group_functions(function_data)

    lines: list[str] = []

    for title, functions in grouped_functions.items():
        if not functions:
            continue
        lines.extend(["", title, "-" * len(title), ""])
        rendered_functions = [f"`{function_name} <{function_data[function_name][0]}>`_" for function_name in functions]
        lines.append(", ".join(rendered_functions))

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

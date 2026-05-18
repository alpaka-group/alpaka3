#!/usr/bin/env python3

import argparse
import pathlib
import json

from var_types import VarType
from get_storage_path import get_storage_path


def get_args():
    """Get command line arguments"""
    parser = argparse.ArgumentParser(description="Write bash variables to a json file.")
    parser.add_argument(
        "--type",
        required=True,
        type=str,
        choices=[e.value for e in VarType],
        help="Variable type",
    )
    parser.add_argument(
        "--name",
        required=True,
        type=str,
        help="Variable name",
    )
    parser.add_argument(
        "--path",
        type=str,
        help="Path to the variable storage file",
    )
    parser.add_argument(
        "values",
        type=str,
        nargs="*",
        help="Variable values",
    )
    return parser.parse_args()


def bash_to_json_type(
    var_type: VarType, values: list[str]
) -> str | list[str] | dict[str, str]:
    """Take a variable in bash notation and return json equivalent.

    Args:
        var_type (VarType): the type of the variable
        values (list[str]): the bash value

    Raises:
        RuntimeError: If type is unknown

    Returns:
        str | list[str] | dict[str, str]: The value
            in a python struct which can be directly mapped to json.
    """

    if var_type == VarType.STRING:
        return "" if len(values) == 0 else values[0]

    if var_type == VarType.ARRAY:
        return values

    if var_type == VarType.MAP:
        return dict(item.split("=", 1) for item in values)

    raise RuntimeError(f"Unknown VarType: {type}")


def write(var_type: VarType, name: str, values: list[str], path: pathlib.Path):
    """Write variable to the storage file.

    Args:
        var_type (VarType): Type of the variable.
        name (str): Name of the variable
        values (list[str]): Value of the variable
        path (pathlib.Path): Path of the storage variable.
    """
    json_value = bash_to_json_type(var_type, values)

    if not path.exists():
        var_storage = {}
    else:
        with open(path, "r", encoding="utf-8") as f:
            var_storage = json.load(f)

    var_storage[name] = json_value
    with open(path, "w", encoding="utf-8") as f:
        json.dump(var_storage, f, indent=4)


if __name__ == "__main__":
    args = get_args()
    if args.path:
        var_storage_path = pathlib.Path(args.path)
    else:
        var_storage_path = get_storage_path()

    write(args.type, args.name, args.values, var_storage_path)

#!/usr/bin/env python3

import argparse
import pathlib
import json
import sys

from var_types import VarType
from get_storage_path import get_storage_path


def get_args():
    """Get command line arguments"""
    parser = argparse.ArgumentParser(description="Write bash variables to a json file.")
    parser.add_argument(
        "--type",
        action="store_true",
        help="Get variable type",
    )
    parser.add_argument(
        "--exist",
        action="store_true",
        help="Check if variable name exist. Application exist with 0, if key exist.",
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
    return parser.parse_args()


def read_config(path: pathlib.Path) -> dict[str, str | list[str] | dict[str, str]]:
    """Load variables from file."""
    with open(path, "r", encoding="utf-8") as f:
        load_var_storage = json.load(f)
    return load_var_storage


def get_type(values: str | list[str] | dict[str, str]) -> VarType:
    """Get the type of a variable.

    Args:
        values (str | list[str] | dict[str, str]): Variable

    Raises:
        RuntimeError: If the type is not known

    Returns:
        VarType: The type
    """
    if isinstance(values, str):
        return VarType.STRING
    if isinstance(values, list):
        return VarType.ARRAY
    if isinstance(values, dict):
        return VarType.MAP

    raise RuntimeError(f"Unknown VarType for value: {values} ({type(values)})")


def json_to_bash_type(values: str | list[str] | dict[str, str]) -> str:
    """Take a value in json notation and return the equivalent bash notation.

    Args:
        values (str | list[str] | dict[str, str]): The variable

    Raises:
        RuntimeError: The type of the variable is not known.

    Returns:
        str: bash notation of the variable.
    """
    val_type: VarType = get_type(values)
    if val_type == VarType.STRING:
        return str(values)
    if val_type == VarType.ARRAY:
        return "(" + " ".join(values) + ")"
    if val_type == VarType.MAP:
        return "(" + " ".join(f"[{kv[0]}]={kv[1]}" for kv in values.items()) + ")"

    raise RuntimeError(f"Unknown VarType for value: {values} ({type(values)})")


if __name__ == "__main__":
    args = get_args()
    if args.path:
        var_storage_path = pathlib.Path(args.path)
    else:
        var_storage_path = get_storage_path()

    var_storage = read_config(var_storage_path)

    if args.exist:
        sys.exit(0 if args.name in var_storage else 1)

    entry = var_storage[args.name]
    if args.type:
        print(get_type(entry).value)
        sys.exit(0)

    print(json_to_bash_type(entry))

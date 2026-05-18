#!/usr/bin/env python3

import pathlib
import os

def get_storage_path() -> pathlib.Path:
    """Return the path to variable storage file depending on the
    environment variables

    Returns:
        pathlib.Path: variable storage file path
    """
    if "VAR_STORAGE_PATH" in os.environ:
        return pathlib.Path(os.environ["VAR_STORAGE_PATH"])

    tmp_folder = pathlib.Path(os.environ.get("TMPDIR", "/tmp"))
    return tmp_folder / "var_storage.json"

if __name__ == "__main__":
    print(get_storage_path())

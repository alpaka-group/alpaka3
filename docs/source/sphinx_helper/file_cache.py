"""Provied functionally to check if a file was changed since last doc build process."""

import re
import hashlib
import json
import pathlib
import subprocess


def get_modified_files(
    cache_file_path: str, path_filter_regex: str = ""
) -> dict[str, str]:
    """Return all files, which changed since last documentation build. If empty, nothing changed.

    Args:
        cache_file_path (str): Name of the cache file.
        path_filter_regex (str, optional): Filter which file paths should be included.
            The paths are relative to the Git project root.
            Defaults to "", which means all files in the repostiory.

    Returns:
        dict[str, str]: Dictionary of files, which changed since last time.
        Key is the relative path to git project root. The value is the sha256 hash of the file.
    """
    cache_file_path_obj = pathlib.Path(cache_file_path)

    old_changes: dict[str, str] = {}
    if cache_file_path_obj.exists():
        with open(cache_file_path_obj, "r", encoding="UTF-8") as f:
            old_changes: dict[str, str] = json.load(f)

    current_changes = get_hashed_files(path_filter_regex)

    def dict_diffs(A: dict, B: dict) -> dict:
        _sentinel = object()

        diffs = {}
        for k in set(A) | set(B):
            a = A.get(k, _sentinel)
            b = B.get(k, _sentinel)
            if a is _sentinel or b is _sentinel or a != b:
                diffs[k] = (
                    None if a is _sentinel else a,
                    None if b is _sentinel else b,
                )
        return diffs

    diff = dict_diffs(old_changes, current_changes)

    with open(cache_file_path_obj, "w", encoding="UTF-8") as f:
        json.dump(current_changes, f, indent=4)

    return diff


def get_hashed_files(path_filter_regex: str = "") -> dict[str, str]:
    """Search for all files which are new or modified basing on git status in the repository.

    Args:
        path_filter_regex (str, optional): Filter which file paths should be included.
            The paths are relative to the Git project root.
            Defaults to "", which means all files in the repostiory.

    Returns:
        dict[str, str]: Dictionary of files.
        Key is the relative path to git project root. The value is the sha256 hash of the file.
    """
    compiled_regex = re.compile(path_filter_regex)

    show_git_toplevel = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        stdout=subprocess.PIPE,
        text=True,
        check=True,
    )
    project_root = pathlib.Path(show_git_toplevel.stdout.strip())

    git_status = subprocess.run(
        ["git", "status", "--porcelain"], stdout=subprocess.PIPE, text=True, check=True
    )

    modified_files = []
    for line in git_status.stdout.strip().split("\n"):
        path = line.strip().split(" ")[1]
        if compiled_regex.match(path):
            if (project_root / pathlib.Path(path)).is_file():
                modified_files.append(path)

    hash_files: dict[str, str] = {}

    for path in modified_files:
        p = project_root / pathlib.Path(path)
        with open(p, "rb") as f:
            digest = hashlib.file_digest(f, "sha256")
        hash_files[path] = digest.hexdigest()

    return hash_files

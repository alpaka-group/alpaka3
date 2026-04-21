"""Utility functions for documentation build."""

import os

def on_rtd() -> bool:
    """Return true, if build is executed on readthedocs.io"""
    return os.environ.get("READTHEDOCS", None) == "True"

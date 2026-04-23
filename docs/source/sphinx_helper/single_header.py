"""Build alpaka single header."""

import os
import sys
import pathlib
import subprocess
from sphinx.util import logging

from .utils import on_rtd
from .file_cache import get_modified_files


def is_generate_single_header(app) -> bool:
    """Check if single header should be build.

    Args:
        app: sphinx doc object

    Returns:
        bool: Return true, if single header should be build.
    """
    logger = logging.getLogger(__name__)

    if on_rtd():
        logger.info("Single Header: create because we are on read the docs.")
        return True

    single_header_path = os.path.abspath(os.path.join(app.builder.outdir))
    if not pathlib.Path(single_header_path).exists():
        logger.info("Single Header: create because single header does not exist.")
        return True

    if "ALPAKA_DOC_SINGLE_HEADER" in os.environ:
        env_value = os.environ["ALPAKA_DOC_SINGLE_HEADER"]
        if env_value in ("1", "ON"):
            logger.info(
                f"Single Header: force build via environment variable ALPAKA_DOC_SINGLE_HEADER={env_value}"
            )
            return True
        if env_value in ("0", "OFF"):
            logger.info(
                f"Single Header: disable build via environment variable ALPAKA_DOC_SINGLE_HEADER={env_value}"
            )
            return False

        logger.error(
            f"Single Header: unknown value for environment variable ALPAKA_DOC_SINGLE_HEADER={env_value}"
        )
        sys.exit(1)

    if not get_modified_files(
        os.path.join(app.builder.outdir, ".single_header_cache.json"), "^include"
    ):
        logger.info("Single Header: skip build because no file was changed")
        return False

    return True


def generate_single_header(app, exception):
    """Build single header.

    Args:
        app: Sphinx doc app object
        exception: Sphinx doc exception object
    """
    if not is_generate_single_header(app):
        return
    # Destination folder relative to conf.py
    single_header_path = os.path.abspath(os.path.join(app.builder.outdir))
    os.makedirs(single_header_path, exist_ok=True)

    # Path to your script
    script_path = os.path.abspath(
        os.path.join(app.srcdir, "..", "..", "script", "create-single-header.sh")
    )

    logger = logging.getLogger(__name__)

    logger.info(f"Generate single header in {single_header_path} (takes some time)")
    # Call the script with the destination folder as argument
    single_header_build = subprocess.run(
        [script_path, single_header_path], stdout=subprocess.PIPE, text=True, check=True
    )

    if single_header_build.returncode != 0:
        logger.error(single_header_build.stderr.strip())

"""Build doxygen Documentation"""

import os
import shutil
import subprocess
import pathlib
import sys
from sphinx.util import logging

from .utils import on_rtd
from .file_cache import get_modified_files


def generate_doxygen(app, exception):
    """Build doxygen documentation.

    Args:
        app: Sphinx doc app object
        exception: Sphinx doc exception object
    """
    if is_generate_doxygen(app):
        build_doxygen(app)


def get_src_dest_paths(app) -> list[tuple[pathlib.Path, pathlib.Path]]:
    """Get the path, where doxygen build the doc (source) and the where it
    needs to copy there (destination). The list contains the user and developer
    build.

    Args:
        app: Sphinx doc app object

    Returns:
        list[tuple[pathlib.Path, pathlib.Path]]: Each entry of the list is a
            (source, destination) tuple.
    """
    # confdir = …/repo-root/docs/source
    conf_dir = pathlib.Path(app.confdir)
    output_dir = pathlib.Path(app.builder.outdir)

    return [
        # USER documentation (docs/doxygen/html)
        (
            (conf_dir / "../doxygen/html")
            .resolve()
            .absolute(),
            (output_dir / "doxygen").absolute(),
        ),
        # DEVELOPER documentation (docs/doxygen_dev/html)
        (
            (conf_dir / "../doxygen_dev/html")
            .resolve()
            .absolute(),
            (output_dir / "doxygen_dev").absolute(),
        ),
    ]


def is_generate_doxygen(app) -> bool:
    """Check if doxygen should be build.

    Args:
        app: sphinx doc object

    Returns:
        bool: Return true, if doxygen should be build.
    """
    logger = logging.getLogger(__name__)

    if on_rtd():
        logger.info("Doxygen: create because we are on read the docs.")
        return True

    if not shutil.which("doxygen"):
        logger.warning(
            "Doxygen: could not find 'doxygen' executable. Skip building doxygen documentation."
        )
        return False

    for _, dest in get_src_dest_paths(app):
        if not dest.exists():
            logger.info(f"Doxygen: build because {dest} does not exist.")
            return True

    if "ALPAKA_DOC_DOXYGEN" in os.environ:
        env_value = os.environ["ALPAKA_DOC_DOXYGEN"]
        if env_value in ("1", "ON"):
            logger.info(
                f"Doxygen: force build via environment variable ALPAKA_DOC_DOXYGEN={env_value}"
            )
            return True
        if env_value in ("0", "OFF"):
            logger.info(
                f"Doxygen: disable build via environment variable ALPAKA_DOC_DOXYGEN={env_value}"
            )
            return False

        logger.error(
            f"Doxygen: unknown value for environment variable ALPAKA_DOC_DOXYGEN={env_value}"
        )
        sys.exit(1)

    if not get_modified_files(
        os.path.join(app.builder.outdir, ".doxygen_cache.json"), "^include"
    ):
        logger.info("Doxygen: skip build because no file was changed")
        return False

    return True


def build_doxygen(app):
    """Run the doxygen build process and copy the rendered doc to the correct position.

    Args:
        app: Sphinx doc app object
    """
    docs_dir = pathlib.Path(app.confdir).parent
    print(docs_dir)

    logger = logging.getLogger(__name__)
    for cmd in (["doxygen"], ["doxygen", "Doxyfile_dev"]):
        logger.info(f"Run {' '.join(cmd)}")
        doxygen_process = subprocess.run(
            cmd, cwd=docs_dir, stdout=subprocess.PIPE, text=True, check=True
        )
        if doxygen_process.stderr:
            logger.warning(doxygen_process.stderr.strip())
        if doxygen_process.returncode != 0:
            logger.error(f"{cmd} failed")
            sys.exit(doxygen_process.returncode)

    # copy user and developer to sphinx doc html output
    for src, dest in get_src_dest_paths(app):
        logger.info(f"copy from {src}\n       to {dest}")
        if src.exists():
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(src, dest)
        else:
            logger.error(f"Doxygen HTML not found at: {src}")
            sys.exit(1)

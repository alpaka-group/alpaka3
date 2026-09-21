"""Build doxygen Documentation"""

import os
import pathlib
import re
import shutil
import subprocess
import sys

from sphinx.util import logging

from .utils import on_rtd
from .file_cache import get_modified_files


def generate_doxygen(app):
    """Build doxygen documentation.

    Args:
        app: Sphinx doc app object
        exception: Sphinx doc exception object
    """
    if is_generate_doxygen(app):
        build_doxygen(app)


def get_dest_paths(app) -> list[pathlib.Path]:
    """Get the paths where doxygen builds the user and developer documentation.

    Args:
        app: Sphinx doc app object

    Returns:
        list[pathlib.Path]: The user and developer documentation output paths.
    """
    output_dir = pathlib.Path(app.builder.outdir)

    return [
        # USER documentation
        (output_dir / "doxygen").absolute(),
        # DEVELOPER documentation
        (output_dir / "doxygen_dev").absolute(),
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
        logger.warning("Doxygen: could not find 'doxygen' executable. Skip building doxygen documentation.")
        return False

    for dest in get_dest_paths(app):
        if not dest.exists():
            logger.info(f"Doxygen: build because {dest} does not exist.")
            return True

    if "ALPAKA_DOC_DOXYGEN" in os.environ:
        env_value = os.environ["ALPAKA_DOC_DOXYGEN"]
        if env_value in ("1", "ON"):
            logger.info(f"Doxygen: force build via environment variable ALPAKA_DOC_DOXYGEN={env_value}")
            return True
        if env_value in ("0", "OFF"):
            logger.info(f"Doxygen: disable build via environment variable ALPAKA_DOC_DOXYGEN={env_value}")
            return False

        logger.error(f"Doxygen: unknown value for environment variable ALPAKA_DOC_DOXYGEN={env_value}")
        sys.exit(1)

    if not get_modified_files(os.path.join(app.builder.outdir, ".doxygen_cache.json"), "^include"):
        logger.info("Doxygen: skip build because no file was changed")
        return False

    return True


def build_doxygen(app):
    """Run the doxygen build process and render the documentation directly
    into the Sphinx output directory.

    Args:
        app: Sphinx doc app object
    """
    docs_dir = pathlib.Path(app.confdir).parent
    print(docs_dir)
    logger = logging.getLogger(__name__)

    destinations = get_dest_paths(app)
    builds = (
        ("Doxyfile", destinations[0]),
        ("Doxyfile_dev", destinations[1]),
    )

    for doxyfile, dest in builds:
        logger.info(f"Run doxygen {doxyfile}")

        if dest.exists():
            shutil.rmtree(dest)
        dest.mkdir(parents=True)

        # Load the original Doxyfile and override the output paths. Setting
        # HTML_OUTPUT to "." prevents doxygen from creating another "html"
        # subdirectory below the destination.
        configuration = (docs_dir / doxyfile).read_text(encoding="utf-8")

        # LIGHT is a value for HTML_COLORSTYLE, while HTML_COLORSTYLE_HUE must
        # be a number between 0 and 359.
        configuration = re.sub(
            r"(?m)^(\s*)HTML_COLORSTYLE_HUE\s*=\s*LIGHT\s*$",
            r"\1HTML_COLORSTYLE = LIGHT",
            configuration,
        )
        configuration += f'\nOUTPUT_DIRECTORY = "{dest}"\nHTML_OUTPUT = .\n'

        doxygen_process = subprocess.run(
            ["doxygen", "-"],
            cwd=docs_dir,
            input=configuration,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        if doxygen_process.stderr:
            logger.warning(doxygen_process.stderr.strip())

        if doxygen_process.returncode != 0:
            logger.error(f"doxygen {doxyfile} failed")
            sys.exit(doxygen_process.returncode)

        if not (dest / "index.html").is_file():
            logger.error(f"Doxygen HTML not found at: {dest}")
            sys.exit(1)

    # The README (used as Doxygen main page) links to files in the repository
    # root (e.g. CITATION.cff) and to images in the docs/ tree. Doxygen does
    # not copy these files into its HTML output, so copy them into every
    # Doxygen output directory (preserving their relative path) to keep the
    # links functional.
    repo_root = docs_dir.parent
    readme_files = {"CITATION.cff": repo_root / "CITATION.cff"}
    for rel_path in (
        "logo/alpaka_401x135.png",
        "images/babelstream-gh200-gpu.svg",
        "images/babelstream-grace-cpu.svg",
    ):
        readme_files[f"docs/{rel_path}"] = docs_dir / rel_path

    for rel_path, source_file in readme_files.items():
        if not source_file.is_file():
            raise FileNotFoundError(f"required README asset not found: {source_file}")

        for dest in destinations:
            target_file = dest / rel_path
            try:
                target_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_file, target_file)
                logger.info(f"copied {source_file.name} to {target_file}")
            except OSError as exc:
                raise RuntimeError(
                    f"could not copy required README asset {source_file} to {target_file}: {exc}"
                ) from exc

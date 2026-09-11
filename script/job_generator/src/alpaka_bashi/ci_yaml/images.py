"""Copyright 2026 Simeon Ehrig
SPDX-License-Identifier: MPL-2.0

Set the image of the GitLab CI test job yaml.
"""

from typing import Any

import bashi
import gitlab
from bashi.globals import (
    ALPAKA_ACC_GPU_CUDA_ENABLE,
    ALPAKA_ACC_GPU_HIP_ENABLE,
    DEVICE_COMPILER,
    GCC,
    HOST_COMPILER,
    OFF_VER,
    UBUNTU,
)
from typeguard import typechecked

from alpaka_bashi.utils import print_warn

GITLAB_IMAGE_CACHE: list[str] | None = None

# is used to display missing image warning one time
image_warning_cache: list[str] = []

CUSTOM_ROCM_IMAGES: dict[str, str] = {
    # The alpaka CI container registry does not provide a prebuilt ROCm 10.0 image yet.
    # Creating the environment on a base alpaka ubuntu container is also not possible due to
    # missing apt packages.
    # Use AMD's official ROCm development image directly from Docker Hub.
    "10.0": "rocm/dev-ubuntu-24.04:10.0.0-full",
}


@typechecked
def get_custom_image_name(combination: bashi.Combination) -> str:
    """Return a custom CI image for combinations not covered by the alpaka CI registry."""
    if combination[ALPAKA_ACC_GPU_HIP_ENABLE].version != OFF_VER:
        return CUSTOM_ROCM_IMAGES.get(
            str(combination[DEVICE_COMPILER].version),
            "",
        )

    return ""


@typechecked
def get_base_image(ubuntu_version) -> str:
    """Return the alpaka CI base image for the given Ubuntu version without container tag."""
    return "registry.hzdr.de/crp/alpaka-group-container/alpaka-ci-ubuntu" + bashi.ubuntu_version_to_string(
        ubuntu_version
    )


@typechecked
def get_images_from_registry(container_version: str) -> list[str]:
    """Returns a list of all container images for a given tag.

    Args:
        container_version (str): Container tag

    Returns:
        List[str]: list of container images
    """
    global GITLAB_IMAGE_CACHE  # pylint: disable=global-statement

    if GITLAB_IMAGE_CACHE is not None:
        return GITLAB_IMAGE_CACHE

    # hard coding the url and project ID is fine, because there is only one
    # container registry
    gl = gitlab.Gitlab(url="https://codebase.helmholtz.cloud")
    project = gl.projects.get("4742")
    repositories = project.repositories.list(get_all=True)

    # uniform data structure
    if not isinstance(repositories, list):
        repositories = [repositories]

    # this process is actual pretty slow
    # maybe it does a HTML request each time if we iterate over repo.tags.list()
    # and get a new tag
    GITLAB_IMAGE_CACHE = [
        tag.attributes["location"]
        for repo in repositories
        for tag in repo.tags.list()
        if tag.attributes["name"] == container_version
    ]

    return GITLAB_IMAGE_CACHE


@typechecked
def get_image_name(combination: bashi.Combination, container_version: str) -> str:
    """Get image name for the given combination. Does not check if the image exist.

    Args:
        combination (bashi.Combination): combination
        container_version (str): container tag

    Returns:
        str: image name
    """
    if custom_image_name := get_custom_image_name(combination):
        return custom_image_name

    image_name = get_base_image(combination[UBUNTU].version)

    cuda_ver = combination[ALPAKA_ACC_GPU_CUDA_ENABLE].version
    if cuda_ver != OFF_VER:
        image_name += f"-cuda{cuda_ver.major}{cuda_ver.minor}"

    if combination[ALPAKA_ACC_GPU_HIP_ENABLE].version != OFF_VER:
        hipcc_ver = combination[DEVICE_COMPILER].version
        image_name += f"-rocm{hipcc_ver.major}.{hipcc_ver.minor}"

    if combination[HOST_COMPILER].name == GCC:
        image_name += "-gcc"

    return f"{image_name}:{container_version}"


@typechecked
def get_existing_image(image_name: str, combination: bashi.Combination, container_version: str) -> str:
    """Check if image exist in the container registry. If not, return container base image for the
    given combination.

    Args:
        image_name (str): image name to be verified
        combination (bashi.Combination): combination
        container_version (str): container tag

    Returns:
        str: If image exist, return `image_name`. Otherwise return base image name.
    """
    if custom_image_name := get_custom_image_name(combination):
        return custom_image_name

    gitlab_images = get_images_from_registry(container_version)
    if image_name in gitlab_images:
        return image_name

    if image_name not in image_warning_cache:
        image_warning_cache.append(image_name)
        print_warn(f"image {image_name} is not available in the registry.")

    return get_base_image(combination[UBUNTU].version) + f":{container_version}"


@typechecked
def set_image(
    job_body: dict[str, Any],
    combination: bashi.Combination,
    container_version: str,
    image_check: bool,
):
    """Set the image of the GitLab CI test job yaml depending on the combination.

    Args:
        job_body (Dict[str, Any]): GitLab CI test job body yaml
        combination (bashi.Combination): combination
    """
    image_name = get_image_name(combination, container_version)

    if image_check:
        image_name = get_existing_image(image_name, combination, container_version)

    job_body["image"] = image_name

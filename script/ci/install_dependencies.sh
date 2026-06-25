#!/usr/bin/env bash

#
# Copyright 2026 Simeon Ehrig
# SPDX-License-Identifier: MPL-2.0
#

# shellcheck source=script/ci/utils/default.sh
source "${APCI_ALPAKA_ROOT}/script/ci/utils/default.sh"
# shellcheck source=script/ci/utils/install_helper_apps.sh
source "${APCI_ALPAKA_ROOT}/script/ci/utils/install_helper_apps.sh"

: "${APCI_DEVICE_COMPILER?'The device compiler must be specified'}"

# disable false positive
# shellcheck disable=SC2218
# SC2218: This function is only defined later. Move the definition up.
script_msg "Install software dependencies (install_dependencies.sh)"

# shellcheck source=script/ci/install/cmake.sh
source "${APCI_ALPAKA_ROOT}/script/ci/install/cmake.sh"
# shellcheck source=script/ci/install/gcc.sh
source "${APCI_ALPAKA_ROOT}/script/ci/install/gcc.sh"
# shellcheck source=script/ci/install/clang.sh
source "${APCI_ALPAKA_ROOT}/script/ci/install/clang.sh"
# shellcheck source=script/ci/install/rocm.sh
source "${APCI_ALPAKA_ROOT}/script/ci/install/rocm.sh"

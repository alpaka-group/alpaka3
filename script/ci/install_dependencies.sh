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

script_msg "Install software dependencies (install_dependencies.sh)"

# shellcheck source=script/ci/install/gcc.sh
source "${APCI_ALPAKA_ROOT}/script/ci/install/gcc.sh"

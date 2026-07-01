#!/usr/bin/env bash

#
# Copyright 2026 Simeon Ehrig
# SPDX-License-Identifier: MPL-2.0
#

: "${APCI_ALPAKA_ROOT?'APCI_ALPAKA_ROOT is not defined. Root directory of the alpaka project'}"

# shellcheck source=script/ci/utils/setup_vars.sh
source "${APCI_ALPAKA_ROOT}/script/ci/utils/setup_vars.sh"
# shellcheck source=script/ci/utils/set.sh
source "${APCI_ALPAKA_ROOT}/script/ci/utils/set.sh"
# shellcheck source=script/ci/utils/color_echo.sh
source "${APCI_ALPAKA_ROOT}/script/ci/utils/color_echo.sh"
# shellcheck source=script/ci/utils/misc.sh
source "${APCI_ALPAKA_ROOT}/script/ci/utils/misc.sh"

# inside the agc-container, the user is root and does not require sudo
# to compatibility to other container, fake the missing sudo command
if ! command -v sudo &>/dev/null; then
    if [[ "$APCI_OS_NAME" == "Linux" ]]; then
        install_msg "sudo"

        lazy_apt_update
        DEBIAN_FRONTEND=noninteractive quiet_run apt install --no-install-recommends -y sudo
    fi
fi

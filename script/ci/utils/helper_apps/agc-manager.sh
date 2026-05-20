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

# the agc-manager only exists in the agc-container
# set alias to false, so each time if we ask the agc-manager if a software is installed, it will
# return false and the installation of software will be triggered
if command -v agc-manager >/dev/null 2>&1; then
    echo_green "found agc-manager"
else
    echo_yellow "install fake agc-manager"
    export PRINT_INSTALL_AGC_MANAGER=true

    echo -e "#!/bin/bash\n\nexit 1" >>agc-manager

    if [[ "$APCI_OS_NAME" = "Linux" ]]; then
        sudo chmod +x agc-manager
        sudo mv agc-manager /usr/bin/agc-manager
    elif [[ "$APCI_OS_NAME" = "Windows" ]]; then
        chmod +x agc-manager
        mv agc-manager /usr/bin
    elif [[ "$APCI_OS_NAME" = "macOS" ]]; then
        sudo chmod +x agc-manager
        sudo mv agc-manager /usr/local/bin
    else
        echo_red "installing agc-manager: " \
            "unknown operation system: ${APCI_OS_NAME}"
        exit 1
    fi
fi

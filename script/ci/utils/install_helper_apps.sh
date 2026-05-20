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
        DEBIAN_FRONTEND=noninteractive apt install -y sudo
    fi
fi

lazy_apt_update
# python3 is required for var_storage.sh
# software-properties-common: 'add-apt-repository' and certificates for wget https download
_helper_apps=(python3 software-properties-common)
install_msg "${_helper_apps[*]}"
DEBIAN_FRONTEND=noninteractive apt install -y "${_helper_apps[@]}"
unset _helper_apps

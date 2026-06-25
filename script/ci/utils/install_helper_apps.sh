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

########################
# install sudo
########################

# shellcheck source=script/ci/utils/helper_apps/sudo.sh
source "${APCI_ALPAKA_ROOT}/script/ci/utils/helper_apps/sudo.sh"

########################
# install agc-manager
########################

# shellcheck source=script/ci/utils/helper_apps/agc-manager.sh
source "${APCI_ALPAKA_ROOT}/script/ci/utils/helper_apps/agc-manager.sh"

########################
# install via apt
########################

lazy_apt_update
# python3 is required for var_storage.sh
# software-properties-common: 'add-apt-repository' and certificates for wget https download
# gnupg2 to add apt keys
# git for CMake fetch content
# make: docs/snippets/CMakeLists.txt requires make, because it
# calls cmake configure as subprocess and does not respect the generator of the parent cmake
# ninja for cmake build
_helper_apps=(python3 software-properties-common wget gnupg2 git make ninja-build)
install_msg "${_helper_apps[*]}"
DEBIAN_FRONTEND=noninteractive apt install -y "${_helper_apps[@]}"
unset _helper_apps

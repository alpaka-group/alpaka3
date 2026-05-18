#!/usr/bin/env bash

#
# Copyright 2026 Simeon Ehrig
# SPDX-License-Identifier: MPL-2.0
#

# source a set of util functions, which are required nearly everywhere

: "${APCI_ALPAKA_ROOT?'APCI_ALPAKA_ROOT is not defined. Root directory of the alpaka project'}"

# shellcheck source=script/ci/utils/set.sh
source "${APCI_ALPAKA_ROOT}/script/ci/utils/set.sh"
# shellcheck source=script/ci/utils/color_echo.sh
source "${APCI_ALPAKA_ROOT}/script/ci/utils/color_echo.sh"
# shellcheck source=script/ci/utils/misc.sh
source "${APCI_ALPAKA_ROOT}/script/ci/utils/misc.sh"

#!/usr/bin/env bash

#
# Copyright 2026 Simeon Ehrig
# SPDX-License-Identifier: MPL-2.0
#

# shellcheck source=script/ci/utils/default.sh
source "${APCI_ALPAKA_ROOT}/script/ci/utils/default.sh"
# shellcheck source=script/ci/utils/sudo.sh
source "${APCI_ALPAKA_ROOT}/script/ci/utils/sudo.sh"

: "${APCI_DEVICE_COMPILER?'The device compiler must be specified'}"

# shellcheck source=script/ci/install/gcc.sh
source "${APCI_ALPAKA_ROOT}/script/ci/install/gcc.sh"

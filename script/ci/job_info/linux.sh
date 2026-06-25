#!/usr/bin/env bash

#
# Copyright 2026 Simeon Ehrig
# SPDX-License-Identifier: MPL-2.0
#

# print all job variables and depending on the OS, how to run the job locally.

echo_yellow "1. Run the container. There are two options.\n
  1.1 Run the container and clone alpaka from the internet.
    - docker run -it ${APCI_IMAGE_NAME} bash
    - apt update && apt install -y git
    - git clone ${APCI_GIT_URL} --depth 1 -b ${APCI_BRANCH_NAME} /alpaka
  1.2 Run the container and mount locally alpaka project.
    - docker run -it -v /path/to/local/alpaka:/alpaka ${APCI_IMAGE_NAME} bash
"

echo_yellow "2. Set the environment variables.\n"

for e in $(env | grep -e "^APCI_"); do
    if [[ "$e" =~ ^APCI_ALPAKA_ROOT ]]; then
        echo_yellow "# use different project path on local system"
        echo_yellow "# CI value: $e"
        echo_yellow "export APCI_ALPAKA_ROOT=/alpaka"
    else
        echo_yellow "export $e"
    fi
done

echo_yellow "\n3. Run the following scripts.\n"

echo_yellow "# install all required software, e.g. compiler, GPU SDKs and more
\$APCI_ALPAKA_ROOT/script/ci/install_dependencies.sh
# run CMake configure
\$APCI_ALPAKA_ROOT/script/ci/configure.sh"

#!/usr/bin/env bash

#
# Copyright 2026 Simeon Ehrig
# SPDX-License-Identifier: MPL-2.0
#

# print all job variables and depending on the OS, how to run the job locally.

echo_yellow "1. Run the container. There two options.
  1.1 Run the container and clone alpaka from the internet
    - docker run -it ${ACPI_IMAGE_NAME} bash
  1.2 Run the container and mount locally alpaka project
    - docker run -it -v /path/to/local/alpaka:/alpaka ${ACPI_IMAGE_NAME} bash
"

for e in $(env | grep -e "^APCI_"); do
    if [[ "$e" =~ ^ACPI_IMAGE_NAME ]]; then
        echo_yellow "# use different project path on local system"
        echo_yellow "# CI value: $e"
        echo_yellow "ACPI_IMAGE_NAME=/alpaka"
    else
        echo_yellow "$e"
    fi
done

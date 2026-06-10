#!/usr/bin/env bash

#
# Copyright 2026 Simeon Ehrig
# SPDX-License-Identifier: MPL-2.0
#

choco uninstall cmake.install
choco install cmake.install --no-progress --version "${APCI_CMAKE}"

APCI_CMAKE_BIN=$(which cmake)
export APCI_CMAKE_BIN

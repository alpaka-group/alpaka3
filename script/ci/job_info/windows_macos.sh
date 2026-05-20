#!/usr/bin/env bash

#
# Copyright 2026 Simeon Ehrig
# SPDX-License-Identifier: MPL-2.0
#

# print all job variables.

for e in $(env | grep -e "^APCI_"); do
    echo_yellow "$e"
done

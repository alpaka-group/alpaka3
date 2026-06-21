#!/usr/bin/env bash
# Copyright 2026 René Widera
# SPDX-License-Identifier: MPL-2.0

failed=0

for f in "$@"; do
    grep -q '^#pragma once$' "$f" || {
        echo "$f: missing #pragma once"
        failed=1
    }
done

exit $failed

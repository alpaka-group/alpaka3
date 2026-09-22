# pylint: disable=missing-docstring

"""Copyright 2026 René Widera
SPDX-License-Identifier: MPL-2.0

Test HIP job execution type annotations.
"""

import unittest

from bashi.globals import DEVICE_COMPILER, HIPCC
from utils import parse_param_value_tuples

from alpaka_bashi.combination_modifier.execution_type import execution_type_hipcc
from alpaka_bashi.globals import JOB_EXECUTION_COMPILE_ONLY_VER, JOB_EXECUTION_RUNTIME_VER, JOB_EXECUTION_TYPE


class TestHipccExecutionType(unittest.TestCase):
    def test_runtime_jobs_are_limited_to_hip_versions_newer_than_6_4(self):
        combinations = [
            parse_param_value_tuples([(DEVICE_COMPILER, HIPCC, "6.3")]),
            parse_param_value_tuples([(DEVICE_COMPILER, HIPCC, "6.4")]),
            parse_param_value_tuples([(DEVICE_COMPILER, HIPCC, "7.0")]),
            parse_param_value_tuples([(DEVICE_COMPILER, HIPCC, "7.0")]),
        ]

        result = execution_type_hipcc(combinations)

        self.assertEqual(result[0][JOB_EXECUTION_TYPE].version, JOB_EXECUTION_COMPILE_ONLY_VER)
        self.assertEqual(result[1][JOB_EXECUTION_TYPE].version, JOB_EXECUTION_COMPILE_ONLY_VER)
        self.assertEqual(result[2][JOB_EXECUTION_TYPE].version, JOB_EXECUTION_RUNTIME_VER)
        self.assertEqual(result[3][JOB_EXECUTION_TYPE].version, JOB_EXECUTION_COMPILE_ONLY_VER)

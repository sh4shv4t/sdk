# Copyright 2026 The Kubeflow Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest

from kubeflow.optimizer.types.search_types import Search
from kubeflow.test.common import FAILED, SUCCESS, TestCase


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="uniform",
            config={"fn": Search.uniform, "args": (0.1, 0.9)},
            expected_output=("0.1", "0.9", "uniform"),
        ),
        TestCase(
            name="loguniform",
            config={"fn": Search.loguniform, "args": (1e-05, 0.1)},
            expected_output=("1e-05", "0.1", "logUniform"),
        ),
        TestCase(
            name="uniform-min-equals-max",
            expected_status=FAILED,
            config={"fn": Search.uniform, "args": (0.5, 0.5)},
            expected_error=ValueError,
        ),
        TestCase(
            name="uniform-min-above-max",
            expected_status=FAILED,
            config={"fn": Search.uniform, "args": (1.0, 0.1)},
            expected_error=ValueError,
        ),
        TestCase(
            name="loguniform-zero-min",
            expected_status=FAILED,
            config={"fn": Search.loguniform, "args": (0, 1)},
            expected_error=ValueError,
        ),
        TestCase(
            name="loguniform-min-above-max",
            expected_status=FAILED,
            config={"fn": Search.loguniform, "args": (0.1, 1e-05)},
            expected_error=ValueError,
        ),
        TestCase(
            name="uniform-nan",
            expected_status=FAILED,
            config={"fn": Search.uniform, "args": (float("nan"), 1.0)},
            expected_error=ValueError,
        ),
        TestCase(
            name="uniform-inf",
            expected_status=FAILED,
            config={"fn": Search.uniform, "args": (0.0, float("inf"))},
            expected_error=ValueError,
        ),
    ],
)
def test_range_search(test_case: TestCase):
    """Test Search.uniform and Search.loguniform bounds validation."""
    print("Executing test:", test_case.name)
    try:
        spec = test_case.config["fn"](*test_case.config["args"])
    except Exception as e:
        assert test_case.expected_status == FAILED
        assert isinstance(e, test_case.expected_error)
    else:
        assert test_case.expected_status == SUCCESS
        space = spec.feasible_space
        assert (space.min, space.max, space.distribution) == test_case.expected_output
    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(name="choice", config={"values": [16, 32]}, expected_output=["16", "32"]),
        TestCase(name="choice-single", config={"values": ["adam"]}, expected_output=["adam"]),
        TestCase(
            name="choice-empty",
            expected_status=FAILED,
            config={"values": []},
            expected_error=ValueError,
        ),
    ],
)
def test_choice_search(test_case: TestCase):
    """Test Search.choice rejects an empty list."""
    print("Executing test:", test_case.name)
    try:
        spec = Search.choice(test_case.config["values"])
    except Exception as e:
        assert test_case.expected_status == FAILED
        assert isinstance(e, test_case.expected_error)
    else:
        assert test_case.expected_status == SUCCESS
        assert spec.feasible_space.list == test_case.expected_output
    print("test execution complete")

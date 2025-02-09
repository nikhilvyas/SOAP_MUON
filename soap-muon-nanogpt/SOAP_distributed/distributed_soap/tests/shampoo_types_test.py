"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

import re
import unittest
from typing import Type, Union

from distributed_soap.shampoo_types import (
    AbstractDataclass,
    DistributedConfig,
)


class InvalidAbstractDataclassInitTest(unittest.TestCase):
    def test_invalid_init(self) -> None:
        for abstract_cls in (AbstractDataclass, DistributedConfig):
            with self.subTest(abstract_cls=abstract_cls), self.assertRaisesRegex(
                TypeError,
                re.escape(
                    f"Cannot instantiate abstract class: {abstract_cls.__name__}."
                ),
            ):
                abstract_cls()
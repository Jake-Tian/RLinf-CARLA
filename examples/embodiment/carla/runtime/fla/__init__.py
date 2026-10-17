# Copyright 2026 The RLinf-CARLA Contributors.
# SPDX-License-Identifier: Apache-2.0

"""Load installed FLA kernels without its unrelated model registrations."""

from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)
__version__ = "0.5.0"

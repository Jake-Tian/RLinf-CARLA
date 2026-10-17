# Copyright 2026 The RLinf-CARLA Contributors.
# SPDX-License-Identifier: Apache-2.0

"""Optional import isolation used by the release acceptance run."""

import importlib.abc
import os
import sys

if os.environ.get("RLINF_VERIFY_STANDALONE") == "1":

    class StandaloneImports(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split(".")[0] in {"verl", "starVLA"}:
                raise ImportError(f"Standalone CARLA acceptance forbids {fullname}")
            return None

    sys.meta_path.insert(0, StandaloneImports())

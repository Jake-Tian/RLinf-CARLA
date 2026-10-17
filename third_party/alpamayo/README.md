# Alpamayo checkpoint compatibility source

This directory contains the 23 model and preprocessing Python files required by the CARLA Alpamayo adapter. They are preserved from the Autopilot-VeRL source commit recorded in [SOURCE.json](SOURCE.json), with original file notices and the [Apache-2.0 license](LICENSE). No additional upstream NOTICE file was found in that checkout.

Install this package with `python -m pip install -e third_party/alpamayo` from the repository root. The CARLA launcher also makes this local package available explicitly, so an external checkout is unnecessary. The `autopilot` package name is retained because checkpoint Hydra targets and internal imports use it. Runtime dependencies are declared in `requirements/embodied/alpamayo.txt` at the repository root.

This is a source subset, not the complete Autopilot/VeRL training framework. Optional upstream SFT helpers retain lazy VeRL imports; the CARLA online path does not call them. The release acceptance blocks VeRL and StarVLA imports and checks worker module origins. It does not validate every historical helper interface.

Model weights, tokenizer assets, normalization metadata and CARLA binaries are not distributed here. This source license is not a claim about model-weight redistribution rights. The model source is kept separate from RLinf-specific adaptation in `rlinf/models/embodiment/alpamayo2/`.

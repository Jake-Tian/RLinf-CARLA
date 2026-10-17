# Copyright 2026 The RLinf-CARLA Contributors.
# SPDX-License-Identifier: Apache-2.0

"""Check checkpoint identity and local model imports before creating workers."""

import hashlib
import json
import os
from pathlib import Path

import transformers
from autopilot.models.alpamayo2 import sft_base_model
from autopilot.utils.dataset import qwen_processor
from transformers import AutoConfig

from rlinf.models.embodiment.alpamayo2 import policy


def main():
    repo = Path(__file__).resolve().parents[3]
    vendor = repo / "third_party/alpamayo"
    modules = {}
    for module in (sft_base_model, qwen_processor, policy):
        path = Path(module.__file__).resolve()
        expected = repo / "rlinf" if module is policy else vendor
        if not path.is_relative_to(expected):
            raise RuntimeError(f"Model import outside release tree: {path}")
        modules[module.__name__] = {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    checkpoint = Path(os.environ["CARLA_SFT_CHECKPOINT"])
    config = AutoConfig.from_pretrained(checkpoint, local_files_only=True)
    if config.model_type != "qwen3_5":
        raise ValueError(f"Expected native qwen3_5, got {config.model_type}")
    metadata = {}
    for name in ("config.json", "reasoning_vla_config.json", "tokenizer_config.json"):
        metadata[name] = hashlib.sha256((checkpoint / name).read_bytes()).hexdigest()
    record = {
        "model_type": config.model_type,
        "transformers": transformers.__version__,
        "model_imports": modules,
        "checkpoint_metadata_sha256": metadata,
    }
    output = Path(os.environ["CARLA_OUTPUT_DIR"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "alpamayo_preflight.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()

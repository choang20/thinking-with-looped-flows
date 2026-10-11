#!/usr/bin/env python3
"""Optional offline Qwen baseline.

The core experiment has no Qwen dependency.  This adapter only loads a model
from a local directory supplied by the user, so it never needs cloud access or
remote code to run.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()

    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    model_path = Path(args.model_path)
    if not model_path.is_dir():
        raise SystemExit(f"local model path does not exist: {model_path}")
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise SystemExit(
            "Install the optional qwen extra only if the model is already local: "
            "python -m pip install -e '.[qwen]'"
        ) from error

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        device_map="cpu",
    )
    inputs = tokenizer(args.prompt, return_tensors="pt")
    output = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
    )
    print(tokenizer.decode(output[0], skip_special_tokens=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

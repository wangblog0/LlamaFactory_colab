#!/usr/bin/env python3
"""
Quantize an exported Hugging Face CausalLM model with llm-compressor AWQ.

Example:
python quantize_awq.py \
  --model-path ./export/qwen3.5-9b-ft-merged \
  --calib-file ./calibration.jsonl \
    --output-path ./export/qwen3.5-9b-ft-merged-awq-ct
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List


def load_calibration_texts(path: Path, max_samples: int, min_chars: int) -> List[str]:
    texts: List[str] = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            if not isinstance(obj, dict):
                continue

            text = obj.get("text")
            if not isinstance(text, str):
                continue

            text = " ".join(text.strip().split())
            if len(text) < min_chars:
                continue

            texts.append(text)
            if len(texts) >= max_samples:
                break

    return texts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AWQ quantization for exported HF model using llm-compressor"
    )
    parser.add_argument("--model-path", required=True, help="Path to merged exported model")
    parser.add_argument("--calib-file", required=True, help="Path to calibration.jsonl")
    parser.add_argument("--output-path", required=True, help="Path to save AWQ model")

    parser.add_argument(
        "--scheme",
        default="W4A16",
        choices=["W4A16", "W4A16_ASYM", "W4AFP8"],
        help="llm-compressor AWQ scheme",
    )
    parser.add_argument(
        "--duo-scaling",
        action="store_true",
        help="Enable AWQ duo scaling (default off for stability)",
    )
    parser.add_argument(
        "--ignore-lm-head",
        action="store_true",
        default=True,
        help="Ignore lm_head during quantization (default true)",
    )
    parser.add_argument(
        "--no-ignore-lm-head",
        action="store_false",
        dest="ignore_lm_head",
        help="Do not ignore lm_head",
    )
    parser.add_argument("--max-calib-samples", type=int, default=512, help="Max calibration samples")
    parser.add_argument("--max-calib-seq-len", type=int, default=512, help="Max calibration seq len")
    parser.add_argument("--min-chars", type=int, default=20, help="Min chars per calibration sample")
    parser.add_argument("--trust-remote-code", action="store_true", default=True, help="Trust remote code")
    parser.add_argument("--no-trust-remote-code", action="store_false", dest="trust_remote_code")

    args = parser.parse_args()

    model_path = Path(args.model_path)
    calib_file = Path(args.calib_file)
    output_path = Path(args.output_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")
    if not calib_file.exists():
        raise FileNotFoundError(f"Calibration file not found: {calib_file}")

    texts = load_calibration_texts(
        path=calib_file,
        max_samples=args.max_calib_samples,
        min_chars=args.min_chars,
    )
    if not texts:
        raise RuntimeError("No valid calibration texts loaded. Check calibration.jsonl format.")

    print(f"Loaded calibration samples: {len(texts)}")

    try:
        from datasets import Dataset
        from llmcompressor import oneshot
        from llmcompressor.modifiers.awq import AWQModifier
        from transformers import AutoModelForCausalLM
        from transformers import AutoTokenizer
    except Exception as exc:
        raise RuntimeError(
            "Missing dependencies. Install with: pip install -U llmcompressor transformers datasets"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        trust_remote_code=args.trust_remote_code,
        use_fast=False,
    )

    ds = Dataset.from_dict({"text": texts})

    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=args.max_calib_seq_len,
            padding=False,
            add_special_tokens=False,
        )

    ds = ds.map(tokenize_fn, batched=True, remove_columns=ds.column_names)

    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=args.trust_remote_code,
    )

    ignore = ["lm_head"] if args.ignore_lm_head else []
    recipe = [
        AWQModifier(
            ignore=ignore,
            scheme=args.scheme,
            targets=["Linear"],
            duo_scaling=args.duo_scaling,
        )
    ]

    print(f"Quant scheme: {args.scheme}")
    print(f"Duo scaling: {args.duo_scaling}")
    print(f"Ignore list: {ignore}")
    print("Starting quantization...")

    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        max_seq_length=args.max_calib_seq_len,
        num_calibration_samples=min(args.max_calib_samples, len(texts)),
    )

    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_path), save_compressed=True)
    tokenizer.save_pretrained(str(output_path))

    print(f"AWQ compressed model saved to: {output_path}")


if __name__ == "__main__":
    main()

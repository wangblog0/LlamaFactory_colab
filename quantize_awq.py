#!/usr/bin/env python3
"""
Quantize an exported Hugging Face CausalLM model to AWQ 4-bit.

Example:
python quantize_awq.py \
  --model-path ./export/qwen3.5-9b-ft-merged \
  --calib-file ./calibration.jsonl \
  --output-path ./export/qwen3.5-9b-ft-merged-awq
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
    parser = argparse.ArgumentParser(description="AWQ quantization for exported HF model")
    parser.add_argument("--model-path", required=True, help="Path to merged exported model")
    parser.add_argument("--calib-file", required=True, help="Path to calibration.jsonl")
    parser.add_argument("--output-path", required=True, help="Path to save AWQ model")

    parser.add_argument("--w-bit", type=int, default=4, help="Weight bit width")
    parser.add_argument("--q-group-size", type=int, default=128, help="AWQ q_group_size")
    parser.add_argument("--max-calib-samples", type=int, default=512, help="Max calibration samples")
    parser.add_argument("--max-calib-seq-len", type=int, default=1024, help="Max calibration seq len")
    parser.add_argument("--min-chars", type=int, default=20, help="Min chars per calibration sample")
    parser.add_argument("--zero-point", action="store_true", default=True, help="Enable zero point")
    parser.add_argument("--no-zero-point", action="store_false", dest="zero_point", help="Disable zero point")
    parser.add_argument("--version", default="GEMM", choices=["GEMM", "GEMV"], help="AWQ kernel version")
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
        from awq import AutoAWQForCausalLM
        from transformers import AutoTokenizer
    except Exception as exc:
        raise RuntimeError(
            "Missing dependencies. Install with: pip install -U autoawq transformers"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        trust_remote_code=args.trust_remote_code,
        use_fast=False,
    )
    model = AutoAWQForCausalLM.from_pretrained(
        str(model_path),
        trust_remote_code=args.trust_remote_code,
    )

    quant_config = {
        "w_bit": args.w_bit,
        "q_group_size": args.q_group_size,
        "zero_point": args.zero_point,
        "version": args.version,
    }

    print(f"Quant config: {quant_config}")
    print("Starting quantization...")

    model.quantize(
        tokenizer,
        quant_config=quant_config,
        calib_data=texts,
        max_calib_samples=len(texts),
        max_calib_seq_len=args.max_calib_seq_len,
    )

    output_path.mkdir(parents=True, exist_ok=True)
    model.save_quantized(str(output_path))
    tokenizer.save_pretrained(str(output_path))

    print(f"AWQ model saved to: {output_path}")


if __name__ == "__main__":
    main()

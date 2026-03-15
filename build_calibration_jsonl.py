#!/usr/bin/env python3
"""
Build calibration.jsonl from common SFT dataset formats.

Supported input:
- .jsonl (one JSON object per line)
- .json (a list of JSON objects, or a dict containing a list)

It tries to extract representative text from common keys:
- text / prompt / query / question
- instruction (+ input)
- conversations / messages (ShareGPT-like)

Output format:
{"text": "..."}
{"text": "..."}
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Iterable, List


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                yield obj
            elif isinstance(obj, list):
                for item in obj:
                    if isinstance(item, dict):
                        yield item


def _iter_json(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    if isinstance(obj, list):
        for item in obj:
            if isinstance(item, dict):
                yield item
        return

    if isinstance(obj, dict):
        # Common wrappers
        for key in ("data", "train", "examples", "items", "records"):
            if key in obj and isinstance(obj[key], list):
                for item in obj[key]:
                    if isinstance(item, dict):
                        yield item
                return
        # Single object fallback
        yield obj


def _normalize_text(text: str, max_chars: int, min_chars: int) -> str | None:
    text = " ".join(text.strip().split())
    if len(text) < min_chars:
        return None
    if len(text) > max_chars:
        text = text[:max_chars]
    return text


def _conversation_to_text(conv: Any) -> str | None:
    if not isinstance(conv, list):
        return None

    chunks: List[str] = []
    for turn in conv:
        if not isinstance(turn, dict):
            continue
        role = str(turn.get("role", turn.get("from", ""))).lower()
        content = turn.get("content", turn.get("value", turn.get("text", "")))
        if not isinstance(content, str):
            continue
        content = content.strip()
        if not content:
            continue
        # Prefer user-side text as calibration data, but keep unknown roles too.
        if role in ("user", "human", "", "system"):
            chunks.append(content)
    if not chunks:
        return None
    return "\n".join(chunks)


def _extract_candidates(record: dict[str, Any]) -> List[str]:
    out: List[str] = []

    # 1) direct text-like keys
    for k in ("text", "prompt", "query", "question"):
        v = record.get(k)
        if isinstance(v, str) and v.strip():
            out.append(v)

    # 2) instruction/input style
    instruction = record.get("instruction")
    inp = record.get("input")
    if isinstance(instruction, str) and instruction.strip():
        if isinstance(inp, str) and inp.strip():
            out.append(f"{instruction}\n{inp}")
        else:
            out.append(instruction)

    # 3) ShareGPT-like
    conv = record.get("conversations")
    conv_text = _conversation_to_text(conv)
    if conv_text:
        out.append(conv_text)

    msgs = record.get("messages")
    msg_text = _conversation_to_text(msgs)
    if msg_text:
        out.append(msg_text)

    return out


def _iter_records(path: Path) -> Iterable[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        yield from _iter_jsonl(path)
    elif suffix == ".json":
        yield from _iter_json(path)
    else:
        raise ValueError(f"Unsupported file type: {path.name}. Use .json or .jsonl")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build calibration.jsonl from SFT data")
    parser.add_argument("--input", required=True, help="Path to source .json or .jsonl dataset")
    parser.add_argument("--output", default="calibration.jsonl", help="Output jsonl path")
    parser.add_argument("--num-samples", type=int, default=512, help="Number of output samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max-chars", type=int, default=4096, help="Max characters per sample")
    parser.add_argument("--min-chars", type=int, default=20, help="Min characters per sample")
    args = parser.parse_args()

    src = Path(args.input)
    dst = Path(args.output)

    if not src.exists():
        raise FileNotFoundError(f"Input file not found: {src}")

    pool: List[str] = []
    seen = set()

    for rec in _iter_records(src):
        for cand in _extract_candidates(rec):
            text = _normalize_text(cand, args.max_chars, args.min_chars)
            if not text:
                continue
            if text in seen:
                continue
            seen.add(text)
            pool.append(text)

    if not pool:
        raise RuntimeError("No valid text extracted. Please check dataset format.")

    random.seed(args.seed)
    random.shuffle(pool)
    selected = pool[: args.num_samples]

    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8") as f:
        for text in selected:
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")

    print(f"Extracted candidates: {len(pool)}")
    print(f"Written samples: {len(selected)}")
    print(f"Output: {dst}")


if __name__ == "__main__":
    main()

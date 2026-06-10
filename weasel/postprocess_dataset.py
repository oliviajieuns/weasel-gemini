"""Build the final WEASEL training subset after greedy selection."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect selected examples from the original dataset, filter long "
            "examples, and optionally subsample to a fixed training budget."
        )
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Original JSON or JSONL dataset indexed by selected dataset indices.",
    )
    parser.add_argument(
        "--selected-indices",
        required=True,
        help="JSON selected-index file from weasel.select_greedy.",
    )
    parser.add_argument("--output", required=True, help="Final dataset output path.")
    parser.add_argument(
        "--max-user-chars",
        type=int,
        default=40000,
        help="Drop examples whose user prompt is longer than this many characters.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=10000,
        help="Uniformly subsample to this many examples after filtering. Use 0 to keep all.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducible subsampling.",
    )
    parser.add_argument(
        "--preserve-order",
        action="store_true",
        help="Preserve dataset-index order after sampling instead of random sample order.",
    )
    parser.add_argument(
        "--dedupe",
        action="store_true",
        help="Remove duplicate selected indices while keeping first occurrence.",
    )
    parser.add_argument(
        "--stats-output",
        default=None,
        help="Optional JSON file with post-processing counts.",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indentation. Use 0 for compact JSON.",
    )
    return parser.parse_args()


def load_jsonl_or_json(path: Path) -> List[Any]:
    with path.open("r", encoding="utf-8") as f:
        first = f.read(1)
        while first and first.isspace():
            first = f.read(1)
        if not first:
            return []
        f.seek(0)
        if first == "[":
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError(f"Expected a JSON list or JSONL file: {path}")
            return data

        # JSONL: iterate physical (\n) lines only. text.splitlines() also splits
        # on U+2028/U+2029/NEL, which are valid UNESCAPED characters inside
        # ensure_ascii=False JSON strings — splitting on them shreds records.
        records: List[Any] = []
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {lineno} of {path}: {exc}"
                ) from exc
        return records


def write_json(data: Any, path: Path, indent: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    actual_indent = None if indent == 0 else indent
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=actual_indent)


def flatten_selected_indices(raw_indices: Any) -> List[int]:
    if not isinstance(raw_indices, list):
        raise ValueError("Selected indices must be a JSON list.")

    flattened: List[int] = []
    for item in raw_indices:
        if isinstance(item, list):
            flattened.extend(int(idx) for idx in item)
        else:
            flattened.append(int(item))
    return flattened


def dedupe_indices(indices: Sequence[int]) -> List[int]:
    seen = set()
    deduped: List[int] = []
    for idx in indices:
        if idx in seen:
            continue
        seen.add(idx)
        deduped.append(idx)
    return deduped


def user_prompt(item: Any) -> str:
    if not isinstance(item, dict):
        return ""
    messages = item.get("messages", [])
    if isinstance(messages, list):
        for message in messages:
            if isinstance(message, dict) and message.get("role") == "user":
                return message.get("content", "")
        try:
            return messages[1].get("content", "")
        except Exception:
            return ""
    return ""


def validate_indices(indices: Sequence[int], dataset_size: int) -> None:
    invalid = [idx for idx in indices if idx < 0 or idx >= dataset_size]
    if invalid:
        preview = invalid[:10]
        raise IndexError(
            f"{len(invalid)} selected indices are outside dataset range "
            f"0-{dataset_size - 1}; first invalid values: {preview}"
        )


def main() -> None:
    args = parse_args()
    dataset = load_jsonl_or_json(Path(args.dataset))
    selected_raw = load_jsonl_or_json(Path(args.selected_indices))
    selected_indices = flatten_selected_indices(selected_raw)

    if args.dedupe:
        selected_indices = dedupe_indices(selected_indices)

    validate_indices(selected_indices, len(dataset))
    selected_items = [dataset[idx] for idx in selected_indices]

    filtered_pairs = [
        (idx, item)
        for idx, item in zip(selected_indices, selected_items)
        if len(user_prompt(item)) <= args.max_user_chars
    ]

    if args.max_examples and len(filtered_pairs) > args.max_examples:
        rng = random.Random(args.seed)
        filtered_pairs = rng.sample(filtered_pairs, args.max_examples)

    if args.preserve_order:
        filtered_pairs = sorted(filtered_pairs, key=lambda pair: pair[0])

    final_data = [item for _, item in filtered_pairs]
    write_json(final_data, Path(args.output), args.indent)

    stats: Dict[str, Any] = {
        "dataset_size": len(dataset),
        "selected_indices": len(selected_indices),
        "after_length_filter": len(
            [
                idx
                for idx in selected_indices
                if len(user_prompt(dataset[idx])) <= args.max_user_chars
            ]
        ),
        "final_size": len(final_data),
        "max_user_chars": args.max_user_chars,
        "max_examples": args.max_examples,
        "seed": args.seed,
        "dedupe": args.dedupe,
        "preserve_order": args.preserve_order,
    }
    print(
        "Postprocessed dataset: "
        f"{stats['selected_indices']} selected -> "
        f"{stats['after_length_filter']} after length filter -> "
        f"{stats['final_size']} final examples"
    )

    if args.stats_output:
        write_json(stats, Path(args.stats_output), args.indent)


if __name__ == "__main__":
    main()

"""Compute WEASEL goal-relevance and pairwise distance metrics.

This script merges the preprocessing work previously split across:

1. remove-redundant-axtree-greedy-algorithm.py
2. phi-score-with-bert-phi-only-obs-and-history.py
3. preprocess-data.py

It intentionally does not select the final subset. It prepares one goal-record
JSON with both importance/phi scores and distance matrices so the downstream
greedy selector can operate from a single artifact.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - keeps the module usable without tqdm
    tqdm = None


GOAL_RE = re.compile(r"##\s*Goal:\s*(.*?)(?=\n##\s|\n#\s|\Z)", re.S)
AXTREE_RE = re.compile(r"##\s*AXTree:\s*(.*?)(?=\n##\s|\n#\s|\Z)", re.S)
OBS_HISTORY_RE = re.compile(
    r"# Observation of current step:\s*(.*?)(?=\n# Action space:|\Z)",
    re.S,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute WEASEL phi and distance metrics in one pass."
    )
    parser.add_argument("--input", required=True, help="Input JSON or JSONL dataset.")
    parser.add_argument(
        "--output",
        required=True,
        help="Output goal-record JSON with phi and distance metrics.",
    )
    parser.add_argument(
        "--augmented-dataset-output",
        default=None,
        help="Optional output dataset with per-example score fields added.",
    )
    parser.add_argument(
        "--model-type",
        default="roberta-large",
        help="BERTScore model used for both importance and diversity.",
    )
    parser.add_argument("--lang", default="en", help="BERTScore language.")
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device for BERTScore. Defaults to cuda when available.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for BERTScore pair scoring.",
    )
    parser.add_argument(
        "--phi-field",
        choices=["axtree", "obs_history", "user_prompt"],
        default="obs_history",
        help="Text compared to the goal for unary relevance and phi.",
    )
    parser.add_argument(
        "--state-field",
        choices=["axtree", "obs_history", "user_prompt"],
        default="axtree",
        help="Text used for pairwise state similarity.",
    )
    parser.add_argument(
        "--response-field",
        choices=["assistant", "reasoning", "action", "assistant_without_think"],
        default="assistant",
        help="Text used for pairwise response similarity.",
    )
    parser.add_argument(
        "--skip-response-distance",
        action="store_true",
        help="Use state-only distance instead of max(state, response).",
    )
    parser.add_argument(
        "--include-prompts",
        action="store_true",
        help="Include original user prompts in the goal-record output.",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indentation. Use 0 for compact JSON.",
    )
    return parser.parse_args()


def load_jsonl_or_json(path: Path) -> List[Dict[str, Any]]:
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
        records: List[Dict[str, Any]] = []
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


def iter_progress(items: Iterable[Any], **kwargs: Any) -> Iterable[Any]:
    if tqdm is None:
        return items
    return tqdm(items, **kwargs)


def get_message_content(
    item: Dict[str, Any],
    role: str,
    fallback_index: Optional[int] = None,
    last: bool = False,
) -> str:
    messages = item.get("messages", [])
    if isinstance(messages, list):
        matches = [
            message.get("content", "")
            for message in messages
            if isinstance(message, dict) and message.get("role") == role
        ]
        if matches:
            return matches[-1] if last else matches[0]

    if fallback_index is not None:
        try:
            return messages[fallback_index].get("content", "")
        except Exception:
            return ""
    return ""


def get_user_prompt(item: Dict[str, Any]) -> str:
    return get_message_content(item, "user", fallback_index=1)


def get_assistant_response(item: Dict[str, Any]) -> str:
    return get_message_content(item, "assistant", fallback_index=2, last=True)


def extract_goal(user_prompt: str) -> str:
    match = GOAL_RE.search(user_prompt or "")
    if not match:
        return "<NO_GOAL_FOUND>"
    return match.group(1).strip()


def extract_axtree(user_prompt: str) -> str:
    matches = [match.group(1).strip() for match in AXTREE_RE.finditer(user_prompt or "")]
    return "\n\n".join(matches)


def extract_obs_history(user_prompt: str) -> str:
    match = OBS_HISTORY_RE.search(user_prompt or "")
    if not match:
        return ""
    return match.group(1).strip()


def extract_tag(text: str, tag: str) -> str:
    match = re.search(rf"<{tag}>(.*?)</{tag}>", text or "", re.S)
    if not match:
        return ""
    return match.group(1).strip()


def extract_action(text: str) -> str:
    return extract_tag(text, "action")


def assistant_without_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()


def text_field(item: Dict[str, Any], field: str) -> str:
    user_prompt = get_user_prompt(item)
    assistant = get_assistant_response(item)

    if field == "axtree":
        return extract_axtree(user_prompt)
    if field == "obs_history":
        return extract_obs_history(user_prompt)
    if field == "user_prompt":
        return user_prompt
    if field == "assistant":
        return assistant
    if field == "reasoning":
        return extract_tag(assistant, "think")
    if field == "action":
        return extract_action(assistant)
    if field == "assistant_without_think":
        return assistant_without_think(assistant)
    raise ValueError(f"Unsupported text field: {field}")


def split_contiguous(indices: Sequence[int]) -> List[List[int]]:
    if not indices:
        return []

    sorted_indices = sorted(indices)
    segments: List[List[int]] = []
    current = [sorted_indices[0]]

    for previous, current_idx in zip(sorted_indices, sorted_indices[1:]):
        if current_idx == previous + 1:
            current.append(current_idx)
        else:
            segments.append(current)
            current = [current_idx]

    segments.append(current)
    return segments


def group_trajectories(data: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    goal_to_indices: Dict[str, List[int]] = defaultdict(list)

    unparseable = 0
    for idx, item in enumerate(data):
        user_prompt = get_user_prompt(item)
        if not user_prompt:
            continue
        goal = extract_goal(user_prompt)
        if not goal or goal == "<NO_GOAL_FOUND>":
            # Never let unparseable goals merge into one shared mega group:
            # pairwise scoring is O(n^2) per group and unrelated tasks would
            # distort each other's selection. Fall back to the source
            # trajectory id (or a singleton per item).
            tid = item.get("_traj_id") if isinstance(item, dict) else None
            goal = (
                f"<NO_GOAL_FOUND:traj#{tid}>" if tid is not None
                else f"<NO_GOAL_FOUND:item#{idx}>"
            )
            unparseable += 1
        goal_to_indices[goal].append(idx)

    if unparseable:
        print(
            f"Warning: {unparseable} datapoints had no parseable '## Goal:'; "
            "grouped by _traj_id/item instead of one shared empty-goal group."
        )

    trajectories: List[Dict[str, Any]] = []
    for goal, indices in goal_to_indices.items():
        trajectories.append(
            {
                "goal": goal,
                "dataset_indices": sorted(indices),
                "segments": split_contiguous(indices),
            }
        )
    return trajectories


def load_bert_scorer(model_type: str, device: Optional[str]):
    try:
        import torch
        from bert_score import BERTScorer
    except ImportError as exc:
        raise RuntimeError(
            "Missing scoring dependencies for BERTScore-based preprocessing."
        ) from exc

    actual_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    return BERTScorer(
        model_type=model_type,
        rescale_with_baseline=False,
        device=actual_device,
    )


def score_pairs(
    scorer: Any,
    candidates: Sequence[str],
    references: Sequence[str],
    batch_size: int,
) -> List[float]:
    if len(candidates) != len(references):
        raise ValueError("candidates and references must have the same length.")
    if not candidates:
        return []

    scores: List[float] = []
    for start in range(0, len(candidates), batch_size):
        end = start + batch_size
        _, _, f1 = scorer.score(
            list(candidates[start:end]),
            list(references[start:end]),
            batch_size=batch_size,
            verbose=False,
        )
        scores.extend(float(value) for value in f1.cpu().tolist())
    return scores


def pairwise_similarity_matrix(
    scorer: Any,
    texts: Sequence[str],
    batch_size: int,
) -> List[List[float]]:
    n = len(texts)
    if n == 0:
        return []
    if n == 1:
        return [[1.0]]

    matrix = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        matrix[i][i] = 1.0

    pair_batch: List[Tuple[int, int]] = []

    def flush_pair_batch() -> None:
        if not pair_batch:
            return

        candidates: List[str] = []
        references: List[str] = []
        for i, j in pair_batch:
            candidates.append(texts[i])
            references.append(texts[j])
            candidates.append(texts[j])
            references.append(texts[i])

        scores = score_pairs(scorer, candidates, references, batch_size)
        cursor = 0
        for i, j in pair_batch:
            similarity = (scores[cursor] + scores[cursor + 1]) / 2.0
            cursor += 2
            matrix[i][j] = similarity
            matrix[j][i] = similarity

        pair_batch.clear()

    for i in range(n):
        for j in range(i + 1, n):
            pair_batch.append((i, j))
            if len(pair_batch) >= batch_size:
                flush_pair_batch()
    flush_pair_batch()

    return matrix


def distance_matrix(
    sims_states: Sequence[Sequence[float]],
    sims_responses: Optional[Sequence[Sequence[float]]] = None,
) -> List[List[float]]:
    n = len(sims_states)
    distances = [[0.0 for _ in range(n)] for _ in range(n)]

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            state_distance = 1.0 - float(sims_states[i][j])
            if sims_responses is None:
                distances[i][j] = state_distance
            else:
                response_distance = 1.0 - float(sims_responses[i][j])
                distances[i][j] = max(state_distance, response_distance)
    return distances


def minmax_normalize(values: Sequence[float]) -> List[float]:
    if not values:
        return []
    min_value = min(values)
    max_value = max(values)
    if max_value == min_value:
        return [0.0 for _ in values]
    return [(value - min_value) / (max_value - min_value) for value in values]


def sum_normalize(values: Sequence[float]) -> List[float]:
    total = sum(values)
    if total == 0:
        return [0.0 for _ in values]
    return [value / total for value in values]


def phi_from_relevance(relevance_scores: Sequence[float]) -> List[float]:
    phi_values: List[float] = []
    previous = 0.0
    for value in relevance_scores:
        phi_values.append(max(0.0, value - previous))
        previous = value
    return phi_values


def compute_relevance_scores(
    scorer: Any,
    goal: str,
    texts: Sequence[str],
    batch_size: int,
) -> List[float]:
    return score_pairs(scorer, list(texts), [goal] * len(texts), batch_size)


def make_empty_scores() -> Dict[str, Any]:
    return {
        "r": None,
        "r_norm": None,
        "phi_raw": None,
        "phi_norm": None,
    }


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    data = load_jsonl_or_json(input_path)
    print(f"Loaded {len(data)} datapoints from {input_path}")

    trajectories = group_trajectories(data)
    num_segments = sum(len(item["segments"]) for item in trajectories)
    largest_segment = max(
        (len(seg) for item in trajectories for seg in item["segments"]), default=0
    )
    print(
        f"Found {len(trajectories)} distinct goals and {num_segments} trajectory "
        f"segments (largest segment: {largest_segment} steps; pairwise scoring "
        "is O(n^2) per segment)"
    )

    scorer = load_bert_scorer(args.model_type, args.device)
    per_item_scores: Dict[int, Dict[str, Any]] = {
        idx: make_empty_scores() for idx in range(len(data))
    }
    goal_records: List[Dict[str, Any]] = []

    for goal_record in iter_progress(
        trajectories, desc="Scoring goals", unit="goal"
    ):
        goal = goal_record["goal"]
        all_indices = goal_record["dataset_indices"]
        all_r_by_index: Dict[int, float] = {}
        all_r_norm_by_index: Dict[int, float] = {}
        all_phi_raw_by_index: Dict[int, float] = {}
        all_phi_norm_by_index: Dict[int, float] = {}
        trajectory_groups: List[Dict[str, Any]] = []

        for segment_indices in goal_record["segments"]:
            segment_items = [data[idx] for idx in segment_indices]
            phi_texts = [text_field(item, args.phi_field) for item in segment_items]
            state_texts = [text_field(item, args.state_field) for item in segment_items]
            response_texts = [
                text_field(item, args.response_field) for item in segment_items
            ]

            r_values = compute_relevance_scores(
                scorer, goal, phi_texts, batch_size=args.batch_size
            )
            r_norm = minmax_normalize(r_values)
            phi_raw = phi_from_relevance(r_values)
            phi_norm = sum_normalize(phi_raw)

            sims_states = pairwise_similarity_matrix(
                scorer,
                state_texts,
                batch_size=args.batch_size,
            )
            sims_responses = None
            if not args.skip_response_distance:
                sims_responses = pairwise_similarity_matrix(
                    scorer,
                    response_texts,
                    batch_size=args.batch_size,
                )
            distances = distance_matrix(sims_states, sims_responses)

            for pos, dataset_idx in enumerate(segment_indices):
                all_r_by_index[dataset_idx] = r_values[pos]
                all_r_norm_by_index[dataset_idx] = r_norm[pos]
                all_phi_raw_by_index[dataset_idx] = phi_raw[pos]
                all_phi_norm_by_index[dataset_idx] = phi_norm[pos]
                per_item_scores[dataset_idx] = {
                    "r": r_values[pos],
                    "r_norm": r_norm[pos],
                    "phi_raw": phi_raw[pos],
                    "phi_norm": phi_norm[pos],
                }

            trajectory_groups.append(
                {
                    "dataset_indices": segment_indices,
                    "bert_scores": r_values,
                    "bert_scores_norm": r_norm,
                    "phi_raw": phi_raw,
                    "phi_norm": phi_norm,
                    "sims_states": sims_states,
                    "sims_responses": sims_responses,
                    "distance_matrix": distances,
                    "state_field": args.state_field,
                    "response_field": None
                    if args.skip_response_distance
                    else args.response_field,
                    "phi_field": args.phi_field,
                }
            )

        record: Dict[str, Any] = {
            "goal": goal,
            "dataset_indices": all_indices,
            "score_metadata": {
                "phi_field": args.phi_field,
                "state_field": args.state_field,
                "response_field": None
                if args.skip_response_distance
                else args.response_field,
                "model_type": args.model_type,
                "lang": args.lang,
            },
            "bert_scores": [all_r_by_index.get(idx, 0.0) for idx in all_indices],
            "bert_scores_norm": [
                all_r_norm_by_index.get(idx, 0.0) for idx in all_indices
            ],
            "phi_scores_raw": [
                all_phi_raw_by_index.get(idx, 0.0) for idx in all_indices
            ],
            "phi_scores_norm": [
                all_phi_norm_by_index.get(idx, 0.0) for idx in all_indices
            ],
            # Backward-compatible names used by the old preprocessing/greedy scripts.
            "bert_scores_obs_history": [
                all_r_by_index.get(idx, 0.0) for idx in all_indices
            ],
            "bert_scores_obs_history_norm": [
                all_r_norm_by_index.get(idx, 0.0) for idx in all_indices
            ],
            "phi_scores_obs_history_raw": [
                all_phi_raw_by_index.get(idx, 0.0) for idx in all_indices
            ],
            "phi_scores_obs_history_norm": [
                all_phi_norm_by_index.get(idx, 0.0) for idx in all_indices
            ],
            "trajectory_groups": trajectory_groups,
        }
        if args.include_prompts:
            record["user_prompts"] = [get_user_prompt(data[idx]) for idx in all_indices]

        goal_records.append(record)

    write_json(goal_records, output_path, args.indent)
    print(f"Saved goal records to {output_path}")

    if args.augmented_dataset_output:
        for idx, item in enumerate(data):
            scores = per_item_scores[idx]
            if scores["r"] is None:
                continue
            item["obs_history_bert_score_r"] = scores["r"]
            item["obs_history_bert_score_r_norm"] = scores["r_norm"]
            item["obs_history_phi_raw"] = scores["phi_raw"]
            item["obs_history_phi_norm"] = scores["phi_norm"]
            item["bert_score_r"] = scores["r"]
            item["bert_score_r_norm"] = scores["r_norm"]
            item["phi_raw"] = scores["phi_raw"]
            item["phi_norm"] = scores["phi_norm"]

        augmented_path = Path(args.augmented_dataset_output)
        write_json(data, augmented_path, args.indent)
        print(f"Saved augmented dataset to {augmented_path}")


if __name__ == "__main__":
    main()

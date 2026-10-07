from __future__ import annotations

import json

from common.data import prompt_messages_from_preference, repo_path


def filter_fitting(tokenizer, rows, max_length):
    """Drop pairs whose prompt alone does not fit max_length. Returns (kept_rows, dropped_indices)."""
    kept, dropped = [], []
    for i, row in enumerate(rows):
        n = len(tokenizer.apply_chat_template(
            prompt_messages_from_preference(row), tokenize=True, add_generation_prompt=True))
        (dropped if n >= max_length else kept).append(i if n >= max_length else row)
    return kept, dropped


def log_dropped(cfg, dataset_path, dropped, n_kept):
    out = repo_path(cfg["results_dir"])
    out.mkdir(parents=True, exist_ok=True)
    stem = str(dataset_path).split("/")[-1].replace(".jsonl", "")
    (out / f"dropped_{stem}.json").write_text(json.dumps({
        "dataset": str(dataset_path),
        "max_sequence_length": int(cfg["max_sequence_length"]),
        "rule": "prompt token length >= max_sequence_length",
        "n_dropped": len(dropped),
        "n_kept": n_kept,
        "dropped_row_indices": dropped,
    }, indent=2))

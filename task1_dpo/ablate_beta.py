from __future__ import annotations

import argparse
import gc
import json

import numpy as np
import torch

from common.data import load_yaml, read_jsonl, repo_path
from common.models import load_policy, load_reward_model, load_tokenizer
from task1_dpo.evaluate import evaluate_model
from task1_dpo.train import run_training


def tag(b):
    return f"beta_{b:.2f}".replace(".", "p")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--betas", help="comma list; default = betas in config")
    ap.add_argument("--wl-samples", type=int, default=8)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    betas = [float(x) for x in args.betas.split(",")] if args.betas else [float(b) for b in cfg["betas"]]
    n = int(cfg["short_ablation_examples"])
    res_dir = repo_path(cfg["results_dir"])
    tok = load_tokenizer(cfg["base_model"])
    rm = rm_tok = None

    for b in betas:
        name = tag(b)
        out_dir = repo_path(f"outputs/task1_dpo/{name}")
        done = (out_dir / "adapter_model.safetensors").exists() and (res_dir / f"train_{name}_summary.json").exists()
        if done:
            print(f"[{name}] training already finished, skipping")
        else:
            print(f"[{name}] training beta={b} on first {n} filtered pairs")
            run_training(args.config, name, None, str(out_dir), beta=b, max_examples=n)
            gc.collect(); torch.cuda.empty_cache()
        if (res_dir / f"eval_{name}.json").exists():
            print(f"[{name}] evaluation already exists, skipping")
            continue
        if rm is None:
            rm, rm_tok = load_reward_model(cfg)
        policy = load_policy(cfg, adapter_path=str(out_dir), trainable=False)
        evaluate_model(policy, tok, cfg, name, b, ["pairs_standard", "gen"], rm, rm_tok, args.wl_samples)
        del policy
        gc.collect(); torch.cuda.empty_cache()

    rows = []
    for b in betas:
        r = json.load(open(res_dir / f"eval_{tag(b)}.json"))
        pairs = read_jsonl(res_dir / f"eval_{tag(b)}_pairs_standard_pairs.jsonl")
        rows.append({"beta": b, "n_train_pairs": n,
                     "pref_acc": r["pairs_standard"]["pref_accuracy"],
                     "median_margin": float(np.median([p["margin"] for p in pairs])),
                     "mean_margin": r["pairs_standard"]["mean_margin"],
                     "dpo_loss_at_own_beta": r["pairs_standard"]["dpo_loss"],
                     "kl_token": r["gen"]["kl_token_mean"], "kl_seq_sum": r["gen"]["kl_seq_sum_mean"],
                     "reward_mean": r["gen"]["reward"]["mean"], "reward_sem": r["gen"]["reward"]["sem"],
                     "len_mean": r["gen"]["length_tokens"]["mean"], "len_std": r["gen"]["length_tokens"]["std"],
                     "frac_truncated": r["gen"]["frac_truncated_at_cap"]})
    (res_dir / "beta_sweep_summary.json").write_text(json.dumps(rows, indent=2))
    for row in rows:
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})


if __name__ == "__main__":
    main()

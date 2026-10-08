from __future__ import annotations

import argparse
import gc
import json

import torch

from common.data import load_yaml, read_jsonl, repo_path
from common.models import load_policy, load_reward_model, load_tokenizer
from common.policy_eval import evaluate_policy, select_prompts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--runs", required=True, help="comma list name=adapter_dir; 'base' = untouched policy")
    ap.add_argument("--n-prompts", type=int, default=100)
    ap.add_argument("--bs", type=int, default=16)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    runs = [tuple(x.split("=", 1)) for x in args.runs.split(",") if x]
    res = repo_path(cfg["results_dir"])
    res.mkdir(parents=True, exist_ok=True)
    tok = load_tokenizer(cfg["base_model"])
    mpl = int(cfg["max_prompt_length"])
    cap = int(cfg.get("eval_max_response_length", cfg["cache_generation_cap"]))  # 768, as in the PPO evaluation
    selected, dropped = select_prompts(tok, read_jsonl(cfg["paths"]["rl_prompt_eval"]), mpl, args.n_prompts)
    ids = [r.get("prompt_id") for _, r in selected]
    (res / "eval_prompt_selection.json").write_text(json.dumps(
        {"n_selected": len(ids), "dropped_row_indices": dropped, "selected_prompt_ids": ids, "max_new_tokens": cap}, indent=1))
    ppo = repo_path("results/task2_ppo/eval_prompt_selection.json")
    if ppo.exists():
        print("prompt set identical to the PPO evaluation:", json.load(open(ppo))["selected_prompt_ids"] == ids)

    todo = [(n, p) for n, p in runs if not (res / f"eval_{n}.json").exists()]
    if todo:
        rm, rm_tok = load_reward_model(cfg)
    for name, path in todo:
        print(f"=== evaluating {name} ({path}) ===", flush=True)
        policy = load_policy(cfg, adapter_path=None if path == "base" else path, trainable=False)
        evaluate_policy(policy, tok, rm, rm_tok, cfg, selected, name, cfg["results_dir"], max_new_tokens=cap,
                        max_prompt_length=mpl, reward_max_length=1280, eos_penalty=0.0, bs=args.bs)
        del policy
        gc.collect()
        torch.cuda.empty_cache()
    print(f"\n{'run':14s} {'reward_raw':>16s} {'KL/token':>9s} {'entropy':>8s} {'len mean+-sd':>14s} {'trunc':>6s}")
    for name, _ in runs:
        r = json.load(open(res / f"eval_{name}.json"))
        rr, ln = r["reward_raw"], r["length_tokens"]
        print(f"{name:14s} {rr['mean']:7.3f}+-{rr['sem']:.3f} {r['kl_token_mean']:9.5f} {r['entropy_token_mean']:8.3f} "
              f"{ln['mean']:6.1f}+-{ln['std']:5.1f} {r['frac_truncated_at_cap']:6.3f}")


if __name__ == "__main__":
    main()

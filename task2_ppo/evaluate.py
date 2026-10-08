from __future__ import annotations

import argparse
import gc
import json
import shutil

import torch

from common.data import load_yaml, read_jsonl, repo_path
from common.models import load_policy, load_reward_model, load_tokenizer
from common.policy_eval import evaluate_policy, select_prompts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--adapter", help="adapter dir, or 'base'")
    ap.add_argument("--name", default="standard")
    ap.add_argument("--runs", help="comma list name=adapter_dir; adapter 'base' = untouched policy")
    ap.add_argument("--n-prompts", type=int, default=100, help="first N usable prompts of the eval pool (same for every condition)")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--sync-dir", help="copy results here after each run (e.g. a Drive folder)")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    runs = [tuple(x.split("=", 1)) for x in args.runs.split(",") if x] if args.runs else [(args.name, args.adapter)]
    res = repo_path(cfg["results_dir"])
    res.mkdir(parents=True, exist_ok=True)
    tok = load_tokenizer(cfg["base_model"])
    mpl = int(cfg["max_prompt_length"])
    selected, dropped = select_prompts(tok, read_jsonl(cfg["paths"]["rl_prompt_eval"]), mpl, args.n_prompts)
    (res / "eval_prompt_selection.json").write_text(json.dumps({
        "n_selected": len(selected), "n_dropped_prompt_too_long": len(dropped), "dropped_row_indices": dropped,
        "selected_row_indices": [i for i, _ in selected], "selected_prompt_ids": [r.get("prompt_id") for _, r in selected],
        "rule": f"first {args.n_prompts} eval-pool rows (file order) whose rendered prompt has <= {mpl} tokens"}, indent=1))
    print(f"{len(selected)} prompts selected, {len(dropped)} dropped (prompt > {mpl} tokens)")

    todo = [(n, p) for n, p in runs if not (res / f"eval_{n}.json").exists()]
    if todo:
        rm, rm_tok = load_reward_model(cfg)
    for name, path in todo:
        print(f"=== evaluating {name} ({path}) ===", flush=True)
        policy = load_policy(cfg, adapter_path=None if path == "base" else path, trainable=False)
        evaluate_policy(policy, tok, rm, rm_tok, cfg, selected, name, cfg["results_dir"],
                        max_new_tokens=int(cfg["eval_max_response_length"]), max_prompt_length=mpl,
                        reward_max_length=int(cfg["reward_max_length"]),
                        eos_penalty=float(cfg.get("missing_eos_penalty", 0.0)), bs=args.bs)
        del policy
        gc.collect(); torch.cuda.empty_cache()
        if args.sync_dir:
            shutil.copytree(res, args.sync_dir, dirs_exist_ok=True)

    print(f"\n{'run':14s} {'reward_eff':>16s} {'reward_raw':>10s} {'KL/token':>9s} {'entropy':>8s} {'len mean±sd':>14s} {'trunc':>6s}")
    for name, _ in runs:
        r = json.load(open(res / f"eval_{name}.json"))
        print(f"{name:14s} {r['reward_effective']['mean']:7.3f}±{r['reward_effective']['sem']:.3f} {r['reward_raw']['mean']:10.3f} "
              f"{r['kl_token_mean']:9.5f} {r['entropy_token_mean']:8.3f}

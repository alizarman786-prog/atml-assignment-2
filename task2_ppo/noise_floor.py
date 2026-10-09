from __future__ import annotations

import json

import torch

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.logging_utils import set_seed
from common.models import load_policy, load_tokenizer, trainable_parameters
from task2_ppo.analyze_clipping import assemble, load_cached_rollouts, policy_logps


def main():
    cfg = load_yaml("configs/ppo.yaml")
    set_seed(int(cfg["seed"]))
    tok = load_tokenizer(cfg["base_model"])
    pool = {r["prompt_id"]: r for r in read_jsonl(cfg["paths"]["rl_prompt_train"]) + read_jsonl(cfg["paths"]["rl_prompt_eval"])}
    cache = load_cached_rollouts(cfg["cached_rollouts"])
    policy = load_policy(cfg, adapter_path=cfg["paths"]["ppo_midpoint_policy"], trainable=True)
    policy.eval()
    for p in trainable_parameters(policy):
        p.data = p.data.float()
    dev = next(policy.parameters()).device
    items = []
    for r in cache:
        rendered = tok.apply_chat_template(prompt_messages(pool[r["prompt_id"]]), tokenize=False, add_generation_prompt=True)
        p = tok(rendered, truncation=True, max_length=int(cfg["max_prompt_length"]))["input_ids"]
        ids = tok(r["response"], add_special_tokens=False)["input_ids"] + ([tok.eos_token_id] if r["terminated_with_eos"] else [])
        items.append(dict(p=p, r=ids[: int(r["old_logprobs"].shape[0])], row=r))
    T = assemble(items, tok, dev)
    out = {mb: policy_logps(policy, T, dev, mb) for mb in (1, 2, 4)}

    def dist(a, b):
        return float(((a - b).abs() * T["mask"]).sum() / T["mask"].sum())

    res = {"description": "mean |logp_recomputed - logp_cached| per response token, PPO midpoint policy, all cached rollouts",
           "n_rollouts": len(items), "n_tokens": int(T["mask"].sum()), "gpu": torch.cuda.get_device_name(0),
           "vs_cache": {f"mb{m}": dist(out[m], T["old"]) for m in (1, 2, 4)},
           "between_own_runs": {"mb1_mb2": dist(out[1], out[2]), "mb2_mb4": dist(out[2], out[4])}}
    path = repo_path(cfg["results_dir"]) / "cache_noise_floor.json"
    path.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

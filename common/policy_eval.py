from __future__ import annotations

import json
import time

import numpy as np
import torch

from common.data import prompt_messages, repo_path
from common.generation import batch_generate, score_reward_pairs
from common.logging_utils import set_seed
from common.models import reference_mode


def select_prompts(tok, rows, max_prompt_length, n):
    """First n rows (file order) whose rendered prompt fits max_prompt_length; identical for every condition."""
    kept, dropped = [], []
    for i, r in enumerate(rows):
        if len(kept) >= n:
            break
        L = len(tok.apply_chat_template(prompt_messages(r), tokenize=True, add_generation_prompt=True))
        if L > max_prompt_length:
            dropped.append(i)
        else:
            kept.append((i, r))
    return kept, dropped


def stats(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n == 0:
        return {"n": 0}
    sd = float(x.std(ddof=1)) if n > 1 else 0.0
    q1, q3 = np.percentile(x, [25, 75])
    return {"n": n, "mean": float(x.mean()), "std": sd, "sem": sd / np.sqrt(n) if n > 1 else 0.0,
            "median": float(np.median(x)), "iqr": float(q3 - q1)}


def _logps(model, seq, attn, pw, resp, entropy=False):
    pos = (attn.cumsum(-1) - 1).clamp(min=0)
    logits = model(input_ids=seq, attention_mask=attn, position_ids=pos, use_cache=False).logits
    logits = logits[:, pw - 1:-1, :][:, : resp.shape[1], :].float()
    logp_all = torch.log_softmax(logits, dim=-1)
    lp = torch.gather(logp_all, -1, resp.unsqueeze(-1)).squeeze(-1)
    ent = -(logp_all.exp() * logp_all).sum(-1) if entropy else None
    return lp, ent


@torch.no_grad()
def evaluate_policy(policy, tok, rm, rm_tok, cfg, selected, name, results_dir, max_new_tokens,
                    max_prompt_length, reward_max_length, eos_penalty=0.0, bs=16, lp_bs=2):
    gen = cfg["generation"]
    rows = [r for _, r in selected]
    prompts = [prompt_messages(r) for r in rows]
    set_seed(int(cfg["seed"]))
    policy.eval()
    order = sorted(range(len(prompts)), key=lambda i: sum(len(m["content"]) for m in prompts[i]))
    recs = [None] * len(prompts)
    t0 = time.time()
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]
        out = batch_generate(policy, tok, [prompts[i] for i in idx], max_prompt_length=max_prompt_length,

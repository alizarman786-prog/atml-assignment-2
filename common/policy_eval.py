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




def _logps(model, seq, attn, pw, resp, entropy=False):
    pos = (attn.cumsum(-1) - 1).clamp(min=0)
    logits = model(input_ids=seq, attention_mask=attn, position_ids=pos, use_cache=False).logits
    logits = logits[:, pw - 1:-1, :][:, : resp.shape[1], :].float()
    logp_all = torch.log_softmax(logits, dim=-1)
    lp = torch.gather(logp_all, -1, resp.unsqueeze(-1)).squeeze(-1)
    ent = -(logp_all.exp() * logp_all).sum(-1) if entropy else None
    return lp, ent


@torch.no_grad()
def generate_all(policy, tok, cfg, prompts, name, max_new_tokens, max_prompt_length, bs, lp_bs):
    gen = cfg["generation"]
    order = sorted(range(len(prompts)), key=lambda i: sum(len(m["content"]) for m in prompts[i]))
    recs = [None] * len(prompts)
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]
        batch = [prompts[i] for i in idx]
        out = batch_generate(
            policy, tok, batch,
            max_prompt_length=max_prompt_length,
            max_new_tokens=max_new_tokens,
            temperature=float(gen["temperature"]),
            top_p=float(gen["top_p"]),
            do_sample=bool(gen["do_sample"]),
        )
        seq, attn = out["sequences"].clone(), out["attention_mask"].clone()
        resp, rmask, pw = out["response_ids"].clone(), out["response_mask"].clone(), out["prompt_width"]
        kl_sum, ent_sum = torch.zeros(len(idx)), torch.zeros(len(idx))
        for j in range(0, len(idx), lp_bs):
            sl = slice(j, j + lp_bs)
            m = rmask[sl]
            Rm = max(int(m.sum(-1).max()), 1)
            sq, at, rs, m = seq[sl][:, :pw + Rm], attn[sl][:, :pw + Rm], resp[sl][:, :Rm], m[:, :Rm]
            pl, ent = _logps(policy, sq, at, pw, rs, entropy=True)
            with reference_mode(policy):
                rl, _ = _logps(policy, sq, at, pw, rs)
            kl_sum[sl] = ((pl - rl) * m).sum(-1).cpu()
            ent_sum[sl] = (ent * m).sum(-1).cpu()
            del pl, rl, ent
        for k, i in enumerate(idx):
            recs[i] = dict(response=out["responses"][k], n_tokens=int(out["response_lengths"][k]),
                           truncated=bool(out["truncated"][k]), terminated=bool(out["terminated_with_eos"][k]),
                           kl_sum=float(kl_sum[k]), entropy_sum=float(ent_sum[k]))
        del out, seq, attn, resp, rmask
        torch.cuda.empty_cache()
        print(f"  [{name}] generated {min(s + bs, len(order))}/{len(order)}", flush=True)
    return recs


@torch.no_grad()
def evaluate_policy(policy, tok, rm, rm_tok, cfg, selected, name, results_dir, max_new_tokens,
                    max_prompt_length, reward_max_length, eos_penalty=0.0, bs=16, lp_bs=2):
    gen = cfg["generation"]
    rows = [r for _, r in selected]
    prompts = [prompt_messages(r) for r in rows]
    set_seed(int(cfg["seed"]))
    policy.eval()
    t0 = time.time()
    recs = generate_all(policy, tok, cfg, prompts, name, max_new_tokens, max_prompt_length, bs, lp_bs)
    for r, row in zip(recs, rows):
        r["prompt_id"] = row.get("prompt_id")
        r["source_index"] = row.get("source_index")

    raw = []
    for s in range(0, len(prompts), 8):
        raw += score_reward_pairs(rm, rm_tok, prompts[s:s + 8], [r["response"] for r in recs[s:s + 8]],
                                  max_length=reward_max_length).cpu().tolist()
    for r, x in zip(recs, raw):
        r["reward_raw"] = float(x)
        r["reward_effective"] = float(x) - eos_penalty * (0.0 if r["terminated"] else 1.0)

    n_tok = max(sum(r["n_tokens"] for r in recs), 1)
    summary = {
        "name": name, "n_prompts": len(recs), "seed": int(cfg["seed"]),
        "max_new_tokens": max_new_tokens, "max_prompt_length": max_prompt_length,
        "missing_eos_penalty": eos_penalty,
        "decoding": {"temperature": gen["temperature"], "top_p": gen["top_p"], "do_sample": gen["do_sample"]},
        "kl_convention": "token-level pooled mean of (logp_policy - logp_ref) over sampled response tokens; "
                         "reference = base model (adapter disabled); kl_seq_sum = per-response summed log-ratio",
        "kl_token_mean": sum(r["kl_sum"] for r in recs) / n_tok,
        "kl_seq_sum": stats([r["kl_sum"] for r in recs]),
        "entropy_token_mean": sum(r["entropy_sum"] for r in recs) / n_tok,
        "reward_raw": stats([r["reward_raw"] for r in recs]),
        "reward_effective": stats([r["reward_effective"] for r in recs]),
        "length_tokens": stats([r["n_tokens"] for r in recs]),
        "frac_truncated_at_cap": float(np.mean([r["truncated"] for r in recs])),
        "eval_wall_clock_s": time.time() - t0,
    }
    res = repo_path(results_dir)
    res.mkdir(parents=True, exist_ok=True)
    with open(res / f"eval_{name}_responses.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (res / f"eval_{name}.json").write_text(json.dumps(summary, indent=2))
    return summary

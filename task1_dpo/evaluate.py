from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from common.data import load_yaml, prompt_messages_from_preference, read_jsonl, repo_path
from common.generation import batch_generate, score_reward_pairs
from common.logging_utils import set_seed
from common.metrics import parse_word_limit, safe_corr, sampled_kl, word_count, word_limit_compliance
from common.models import load_policy, load_reward_model, load_tokenizer, reference_mode
from task1_dpo.dpo import dpo_loss
from task1_dpo.dpo_loop import sequence_logps
from task1_dpo.filtering import filter_fitting, log_dropped
from task1_dpo.train import make_collate

ALL_SETS = ["pairs_standard", "pairs_length", "gen", "wordlimit"]


def _stats(x):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return {"n": 0}
    q1, q3 = np.percentile(x, [25, 75])
    return {"n": int(len(x)), "mean": float(x.mean()), "std": float(x.std(ddof=1)) if len(x) > 1 else 0.0,
            "sem": float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0,
            "median": float(np.median(x)), "iqr": float(q3 - q1)}


# ---------------------------------------------------------------- held-out preference pairs
@torch.no_grad()
def pair_metrics(model, tokenizer, rows, cfg, beta, bs=4):
    collate = make_collate(tokenizer, int(cfg["max_sequence_length"]))
    device = next(model.parameters()).device
    pc, pr, rc, rr, nc, nr = [], [], [], [], [], []
    for i in range(0, len(rows), bs):
        chosen, rejected = collate(rows[i:i + bs])
        a, n1 = sequence_logps(model, chosen, device)
        b, n2 = sequence_logps(model, rejected, device)
        with reference_mode(model):
            c, _ = sequence_logps(model, chosen, device)
            d, _ = sequence_logps(model, rejected, device)
        for lst, v in zip((pc, pr, rc, rr, nc, nr), (a, b, c, d, n1, n2)):
            lst.append(v.cpu())
    pc, pr, rc, rr, nc, nr = [torch.cat(x) for x in (pc, pr, rc, rr, nc, nr)]
    loss, d = dpo_loss(pc, pr, rc, rr, beta)
    margin = (pc - rc) - (pr - rr)
    per_pair = []
    for k, row in enumerate(rows):
        per_pair.append({"prompt_id": row.get("prompt_id"), "stratum": row.get("length_stratum"),
                         "margin": float(margin[k]), "chosen_logratio": float(pc[k] - rc[k]),
                         "rejected_logratio": float(pr[k] - rr[k]),
                         "chosen_tokens": int(nc[k]), "rejected_tokens": int(nr[k])})
    agg = {"n_pairs": len(rows), "beta": beta, "dpo_loss": float(loss), "pref_accuracy": float(d["preference_accuracy"]),
           "mean_margin": float(margin.mean()), "mean_chosen_logratio": float((pc - rc).mean()),
           "mean_rejected_logratio": float((pr - rr).mean()),
           "corr_margin_vs_length_diff": safe_corr(margin.numpy(), (nc - nr).numpy())}
    strata = sorted({p["stratum"] for p in per_pair if p["stratum"] is not None})
    if strata:
        agg["per_stratum"] = {}
        for s in strata:
            m = np.array([p["margin"] for p in per_pair if p["stratum"] == s])
            agg["per_stratum"][s] = {"n": int(len(m)), "pref_accuracy": float((m > 0).mean()), "mean_margin": float(m.mean())}
    return agg, per_pair


# ---------------------------------------------------------------- generation, KL, reward
def token_logps(model, seq, attn, prompt_width, resp_ids):
    # position_ids from the attention mask so left-padded rows are scored at their true positions
    pos = (attn.cumsum(-1) - 1).clamp(min=0)
    logits = model(input_ids=seq, attention_mask=attn, position_ids=pos, use_cache=False).logits
    logits = logits[:, prompt_width - 1:-1, :][:, : resp_ids.shape[1], :]
    logp = torch.log_softmax(logits.float(), dim=-1)
    return torch.gather(logp, -1, resp_ids.unsqueeze(-1)).squeeze(-1)


@torch.no_grad()
def run_generation(model, tok, prompts, cfg, bs=16, lp_bs=4, with_kl=True):
    gen = cfg["generation"]
    set_seed(int(cfg["seed"]))
    order = sorted(range(len(prompts)), key=lambda i: sum(len(m["content"]) for m in prompts[i]))
    recs = [None] * len(prompts)
    kl_vals, kl_toks = [], []
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]
        out = batch_generate(model, tok, [prompts[i] for i in idx],
                             max_prompt_length=int(cfg["max_sequence_length"]),
                             max_new_tokens=int(cfg["max_generation_tokens"]),
                             temperature=float(gen["temperature"]), top_p=float(gen["top_p"]),
                             do_sample=bool(gen["do_sample"]))
        seq, attn = out["sequences"].clone(), out["attention_mask"].clone()
        resp, rmask, pw = out["response_ids"].clone(), out["response_mask"].clone(), out["prompt_width"]
        kl_sum = torch.zeros(len(idx))
        if with_kl:
            for j in range(0, len(idx), lp_bs):
                sl = slice(j, j + lp_bs)
                pl = token_logps(model, seq[sl], attn[sl], pw, resp[sl])
                with reference_mode(model):
                    rl = token_logps(model, seq[sl], attn[sl], pw, resp[sl])
                m = rmask[sl]
                kl_vals.append(float(sampled_kl(pl, rl, m)))
                kl_toks.append(float(m.sum()))
                kl_sum[sl] = ((pl - rl) * m).sum(-1).cpu()
        for k, i in enumerate(idx):
            recs[i] = {"response": out["responses"][k], "n_tokens": int(out["response_lengths"][k]),
                       "truncated": bool(out["truncated"][k]), "terminated": bool(out["terminated_with_eos"][k]),
                       "kl_sum": float(kl_sum[k])}
        del out, seq, attn, resp, rmask
        torch.cuda.empty_cache()
        print(f"  generated {min(s + bs, len(order))}/{len(order)}", flush=True)
    return recs, kl_vals, kl_toks


def score_rewards(rm, rm_tok, prompts, responses, bs=8):
    out = []
    for i in range(0, len(prompts), bs):
        out += score_reward_pairs(rm, rm_tok, prompts[i:i + bs], responses[i:i + bs]).cpu().tolist()
    return out


def _dump_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------- orchestration
def evaluate_model(policy, tok, cfg, name, beta, sets=ALL_SETS, rm=None, rm_tok=None,
                   wl_samples=8, limit=None, is_base=False):
    res_dir = repo_path(cfg["results_dir"])
    res_dir.mkdir(parents=True, exist_ok=True)
    gen = cfg["generation"]
    out = {"name": name, "loss_beta": beta, "seed": int(cfg["seed"]), "is_base": is_base, "sets": sets, "limit": limit,
           "decoding": {"temperature": gen["temperature"], "top_p": gen["top_p"], "do_sample": gen["do_sample"],
                        "max_new_tokens": cfg["max_generation_tokens"]},
           "kl_convention": "token-level mean of (logp_policy - logp_ref) pooled over all sampled response tokens "
                            "(course sampled_kl, weighted by token count); kl_seq_sum_mean = mean over responses of the summed log-ratio",
           "max_sequence_length": cfg["max_sequence_length"]}
    t0 = time.time()

    def load_filtered(key):
        path = cfg["paths"][key]
        allrows = read_jsonl(path)
        kept, dropped = filter_fitting(tok, allrows, int(cfg["max_sequence_length"]))
        dset = set(dropped)
        kept_idx = [i for i in range(len(allrows)) if i not in dset]
        if limit is None:
            log_dropped(cfg, path, dropped, len(kept))
        else:
            kept, kept_idx = kept[:limit], kept_idx[:limit]
        return allrows, kept, kept_idx, dropped

    for key, data_key in (("pairs_standard", "dpo_standard_eval"), ("pairs_length", "dpo_length_eval")):
        if key not in sets or is_base:
            continue
        print(f"[{name}] {key}", flush=True)
        allrows, kept, kept_idx, dropped = load_filtered(data_key)
        agg, per_pair = pair_metrics(policy, tok, kept, cfg, beta)
        for p, i in zip(per_pair, kept_idx):
            p["row_index"] = i
        agg["n_total_in_file"], agg["n_dropped_prompt_too_long"] = len(allrows), len(dropped)
        if kept and kept[0].get("length_stratum") is not None:
            from collections import Counter
            agg["stratum_counts_before_filter"] = dict(Counter(r["length_stratum"] for r in allrows))
        out[key] = agg
        _dump_jsonl(res_dir / f"eval_{name}_{key}_pairs.jsonl", per_pair)

    if "gen" in sets:
        print(f"[{name}] gen", flush=True)
        allrows, kept, kept_idx, dropped = load_filtered("dpo_standard_eval")
        prompts = [prompt_messages_from_preference(r) for r in kept]
        recs, klv, klt = run_generation(policy, tok, prompts, cfg, with_kl=True)
        rewards = score_rewards(rm, rm_tok, prompts, [r["response"] for r in recs])
        for r, rw, row, i in zip(recs, rewards, kept, kept_idx):
            r.update(reward=float(rw), prompt_id=row.get("prompt_id"), row_index=i)
        out["gen"] = {"n_prompts": len(recs),
                      "kl_token_mean": float(np.average(klv, weights=klt)),
                      "kl_seq_sum_mean": float(np.mean([r["kl_sum"] for r in recs])),
                      "reward": _stats(rewards),
                      "length_tokens": _stats([r["n_tokens"] for r in recs]),
                      "frac_truncated_at_cap": float(np.mean([r["truncated"] for r in recs]))}
        _dump_jsonl(res_dir / f"eval_{name}_gen_responses.jsonl", recs)

    if "wordlimit" in sets:
        print(f"[{name}] wordlimit", flush=True)
        rows = read_jsonl(cfg["paths"]["word_limit_prompts"])
        if limit:
            rows = rows[:limit]
        prompts, meta = [], []
        for r in rows:
            for s in range(wl_samples):
                prompts.append(r["messages"])
                meta.append((r["prompt_id"], s))
        recs, _, _ = run_generation(policy, tok, prompts, cfg, with_kl=False)
        for rec, (pid, s), p in zip(recs, meta, prompts):
            text = p[-1]["content"]
            rec.update(prompt_id=pid, sample=s, words=word_count(rec["response"]),
                       limit=parse_word_limit(text), compliant=word_limit_compliance(text, rec["response"]))
        parsed = [r for r in recs if r["limit"] is not None]
        per_prompt = {}
        for r in parsed:
            per_prompt.setdefault(r["prompt_id"], []).append(r["compliant"])
        out["wordlimit"] = {"n_prompts": len(rows), "samples_per_prompt": wl_samples,
                            "n_prompts_with_parsed_limit": len(per_prompt),
                            "compliance_rate": float(np.mean([r["compliant"] for r in parsed])) if parsed else None,
                            "per_prompt_compliance": {k: float(np.mean(v)) for k, v in per_prompt.items()},
                            "words": _stats([r["words"] for r in recs]),
                            "length_tokens": _stats([r["n_tokens"] for r in recs])}
        _dump_jsonl(res_dir / f"eval_{name}_wordlimit_responses.jsonl", recs)

    out["eval_wall_clock_s"] = time.time() - t0
    (res_dir / f"eval_{name}.json").write_text(json.dumps(out, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--adapter", required=True, help="adapter directory, or 'base' for the untouched policy")
    ap.add_argument("--name", default="standard")
    ap.add_argument("--beta", type=float, help="beta used for the held-out DPO loss (default: config beta)")
    ap.add_argument("--sets", default=",".join(ALL_SETS))
    ap.add_argument("--wl-samples", type=int, default=8)
    ap.add_argument("--limit", type=int, help="SMOKE TESTS ONLY: first N rows of each set")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    is_base = args.adapter == "base"
    sets = [s for s in args.sets.split(",") if s]
    tok = load_tokenizer(cfg["base_model"])
    policy = load_policy(cfg, adapter_path=None if is_base else args.adapter, trainable=False)
    rm = rm_tok = None
    if "gen" in sets:
        rm, rm_tok = load_reward_model(cfg)
    beta = float(cfg["beta"] if args.beta is None else args.beta)
    res = evaluate_model(policy, tok, cfg, args.name, beta, sets, rm, rm_tok, args.wl_samples, args.limit, is_base)
    print(json.dumps({k: v for k, v in res.items() if k in sets or k == "eval_wall_clock_s"}, indent=2))


if __name__ == "__main__":
    main()

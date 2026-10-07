from __future__ import annotations

import argparse
import gc
import json

import numpy as np
import pandas as pd

from common.data import load_yaml, read_jsonl, repo_path
from common.models import load_policy, load_reward_model, load_tokenizer
from task1_dpo.evaluate import ALL_SETS, evaluate_model
from task1_dpo.train import run_training

NAME = "length_balanced"


def paired(a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    se = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("nan")
    return {"mean_diff": float(d.mean()), "se": se, "z": float(d.mean() / se) if se and se > 0 else None, "n": int(len(d))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--wl-samples", type=int, default=8)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    res = repo_path(cfg["results_dir"])
    out_dir = repo_path(cfg["length_output"])

    if (out_dir / "adapter_model.safetensors").exists() and (res / f"train_{NAME}_summary.json").exists():
        print(f"[{NAME}] training already finished, skipping")
    else:
        print(f"[{NAME}] training on {cfg['paths']['dpo_length_train']}")
        run_training(args.config, NAME, cfg["paths"]["dpo_length_train"], str(out_dir))
        gc.collect()
        import torch; torch.cuda.empty_cache()

    if not (res / f"eval_{NAME}.json").exists():
        tok = load_tokenizer(cfg["base_model"])
        rm, rm_tok = load_reward_model(cfg)
        policy = load_policy(cfg, adapter_path=str(out_dir), trainable=False)
        evaluate_model(policy, tok, cfg, NAME, float(cfg["beta"]), ALL_SETS, rm, rm_tok, args.wl_samples)
        del policy
    else:
        print(f"[{NAME}] evaluation already exists, skipping")

    out = {}
    # ---- per-stratum pair accuracy, standard vs length-balanced (same filtered pairs, paired SE)
    S = pd.DataFrame(read_jsonl(res / "eval_standard_pairs_length_pairs.jsonl")).set_index("row_index")
    L = pd.DataFrame(read_jsonl(res / f"eval_{NAME}_pairs_length_pairs.jsonl")).set_index("row_index")
    assert (S.index == L.index).all()
    out["pairs"] = {}
    for s in sorted(S.stratum.unique()) + ["ALL"]:
        idx = S.index if s == "ALL" else S.index[S.stratum == s]
        cs, cl = (S.loc[idx, "margin"] > 0).astype(float).values, (L.loc[idx, "margin"] > 0).astype(float).values
        out["pairs"][s] = {"n": int(len(idx)), "acc_standard": float(cs.mean()), "acc_balanced": float(cl.mean()),
                           "acc_diff_balanced_minus_standard": paired(cl, cs),
                           "mean_margin_standard": float(S.loc[idx, "margin"].mean()),
                           "mean_margin_balanced": float(L.loc[idx, "margin"].mean())}
    for tag, df in (("standard", S), ("balanced", L)):
        dl = df.chosen_tokens - df.rejected_tokens
        out[f"corr_margin_vs_lengthdiff_{tag}"] = float(df.margin.corr(dl))
        out[f"corr_chosen_logratio_vs_tokens_{tag}"] = float(df.chosen_logratio.corr(df.chosen_tokens))
        out[f"corr_rejected_logratio_vs_tokens_{tag}"] = float(df.rejected_logratio.corr(df.rejected_tokens))

    # ---- generation on the standard held-out prompts: reward and length
    G = {n: pd.DataFrame(read_jsonl(res / f"eval_{n}_gen_responses.jsonl")).set_index("prompt_id") for n in ("base", "standard", NAME)}
    out["gen"] = {}
    for n, g in G.items():
        out["gen"][n] = {"reward_mean": float(g.reward.mean()), "reward_sem": float(g.reward.sem()),
                         "len_mean": float(g.n_tokens.mean()), "len_std": float(g.n_tokens.std()),
                         "frac_truncated": float(g.truncated.mean())}
    for other in ("base", "standard"):
        j = G[NAME].join(G[other], lsuffix="_bal", rsuffix="_oth", how="inner")
        out["gen"][f"paired_vs_{other}"] = {"reward": paired(j.reward_bal, j.reward_oth), "length": paired(j.n_tokens_bal, j.n_tokens_oth)}

    # ---- word-limit compliance on the common prompt set
    W = {n: pd.DataFrame(read_jsonl(res / f"eval_{n}_wordlimit_responses.jsonl")).set_index(["prompt_id", "sample"]) for n in ("base", "standard", NAME)}
    out["wordlimit"] = {}
    for n, w in W.items():
        out["wordlimit"][n] = {"compliance": float(w.compliant.mean()), "words_mean": float(w.words.mean()), "tokens_mean": float(w.n_tokens.mean())}
    for other in ("base", "standard"):
        j = W[NAME].join(W[other], lsuffix="_bal", rsuffix="_oth", how="inner")
        out["wordlimit"][f"paired_vs_{other}"] = paired(j.compliant_bal, j.compliant_oth)

    (res / "length_analysis.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

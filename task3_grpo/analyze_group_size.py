from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from common.data import load_yaml, read_jsonl, repo_path

EPS = 1e-6  # same tolerance as group_relative_advantages


def load(cfg):
    df = pd.DataFrame(read_jsonl(cfg["group_cache"]))
    df["reward"] = df["reward"].astype(float)
    df["generation_index"] = df["generation_index"].astype(int)
    df["ntok"] = df["completion_tokens"].astype(int)
    df["clipped"] = df["ntok"] >= int(cfg["max_completion_length"])  # would be masked under the training cap (512)
    return df.sort_values(["prompt_id", "generation_index"]).reset_index(drop=True)


def group_metrics(R, C):
    """R, C: [n_groups, K] rewards and masked flags. Advantages use all K completions (as in grpo.py);
    masked completions are then removed from the loss, as in the training loop."""
    mu, sd = R.mean(1, keepdims=True), R.std(1, keepdims=True)  # population std, as in grpo.py
    informative = sd[:, 0] > EPS
    adv = np.where(informative[:, None], (R - mu) / (sd + EPS), 0.0)
    unmasked = ~C
    trainable = (unmasked & (np.abs(adv) > 0)).any(1)
    return {"n_groups": int(len(R)), "informative_rate": float(informative.mean()),
            "mean_group_std": float(sd.mean()), "mean_group_var": float((sd ** 2).mean()),
            "adv_var_all_completions": float(adv.var()), "frac_completions_masked": float(C.mean()),
            "trainable_group_rate": float(trainable.mean()),
            "mean_abs_adv_unmasked": float(np.abs(adv)[unmasked].mean()) if unmasked.any() else float("nan")}


def regroup(df, K):
    out = {}
    for pid, g in df.groupby("prompt_id", sort=True):
        n = len(g) // K * K
        out[pid] = (g["reward"].values[:n].reshape(-1, K), g["clipped"].values[:n].reshape(-1, K))
    return out


def metrics_for(parts, pids):
    R = np.concatenate([parts[p][0] for p in pids])
    C = np.concatenate([parts[p][1] for p in pids])
    return group_metrics(R, C)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--boot", type=int, default=1000)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    df = load(cfg)
    counts = df.groupby("prompt_id").size()
    print(f"{len(df)} completions, {len(counts)} prompts, completions per prompt: {sorted(set(counts))}")
    Ks = [int(k) for k in cfg["group_sizes"]]
    pm = df.groupby("prompt_id")["reward"].mean().sort_values()
    order, t = list(pm.index), len(pm) // 3
    bins = {"hard": order[:t], "medium": order[t:len(order) - t], "easy": order[len(order) - t:]}
    parts = {K: regroup(df, K) for K in Ks}
    rng = np.random.RandomState(int(cfg["seed"]))
    flag = df["clipped_at_max"].map(lambda x: str(x).lower() == "true")
    out = {"total_completions": int(len(df)), "n_prompts": int(len(counts)), "completions_per_prompt": sorted(set(int(c) for c in counts)),
           "mask_rule": f"completion_tokens >= {cfg['max_completion_length']} (training cap), not the 768 cache cap",
           "frac_completions_ge_768_flag": float(flag.mean()),
           "difficulty_bins": {"rule": "prompt mean reward over all cached completions, terciles (hard=lowest)",
                               "prompts": {b: list(p) for b, p in bins.items()},
                               "mean_reward": {b: float(pm[p].mean()) for b, p in bins.items()}}}
    rows = []
    for K in Ks:
        res = {"all": metrics_for(parts[K], order)}
        res.update({b: metrics_for(parts[K], p) for b, p in bins.items()})
        boots = [metrics_for(parts[K], [order[i] for i in rng.randint(0, len(order), len(order))]) for _ in range(args.boot)]
        for key in ("informative_rate", "mean_group_std", "trainable_group_rate"):
            v = [b[key] for b in boots]
            res["all"][key + "_ci95"] = [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
        out[f"K{K}"] = res
        for b in ("all", "hard", "medium", "easy"):
            m = res[b]
            rows.append(dict(K=K, bin=b, groups=m["n_groups"], informative=m["informative_rate"], std=m["mean_group_std"],
                             var=m["mean_group_var"], adv_var=m["adv_var_all_completions"], masked=m["frac_completions_masked"],
                             trainable=m["trainable_group_rate"]))
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    for K in Ks:
        a = out[f"K{K}"]["all"]
        print(f"K={K}: informative 95% CI {np.round(a['informative_rate_ci95'], 3)}, std CI {np.round(a['mean_group_std_ci95'], 3)}, trainable CI {np.round(a['trainable_group_rate_ci95'], 3)}")
    p = repo_path(cfg["results_dir"]); p.mkdir(parents=True, exist_ok=True)
    (p / "group_size_study.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

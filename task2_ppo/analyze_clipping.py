from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.logging_utils import set_seed
from common.models import load_policy, load_tokenizer, trainable_parameters
from task2_ppo.ppo import compute_gae, normalize_advantages, ppo_policy_loss, shaped_rewards
from task2_ppo.ppo_loop import _load, _state, token_logps


def load_cached_rollouts(path):
    rows = torch.load(repo_path(path), map_location="cpu", weights_only=False)
    if not isinstance(rows, list) or not rows:
        raise ValueError("Expected a non-empty list in the supplied PPO rollout cache")
    normalized = []
    for row in rows:
        row = dict(row)
        if "old_logprobs" not in row and "old_policy_logprobs" in row:
            row["old_logprobs"] = row["old_policy_logprobs"]
        if "ref_logprobs" not in row and "reference_logprobs" in row:
            row["ref_logprobs"] = row["reference_logprobs"]
        normalized.append(row)
    required = {"source_index", "response", "old_logprobs", "ref_logprobs"}
    if not required.issubset(normalized[0]):
        raise ValueError(f"Unexpected PPO cache schema; need at least {sorted(required)}")
    return normalized


def assemble(items, tok, device):
    pw = max(len(it["p"]) for it in items)
    R = max(len(it["r"]) for it in items)
    B = len(items)
    seq = torch.full((B, pw + R), tok.pad_token_id, dtype=torch.long)
    attn = torch.zeros((B, pw + R), dtype=torch.long)
    old, ref, val, mask = (torch.zeros(B, R) for _ in range(4))
    for b, it in enumerate(items):
        lp, n = len(it["p"]), len(it["r"])
        seq[b, pw - lp:pw] = torch.tensor(it["p"]); attn[b, pw - lp:pw] = 1
        seq[b, pw:pw + n] = torch.tensor(it["r"]); attn[b, pw:pw + n] = 1
        old[b, :n] = it["row"]["old_logprobs"].float(); ref[b, :n] = it["row"]["ref_logprobs"].float()
        val[b, :n] = it["row"]["values"].float(); mask[b, :n] = 1
    task_r = torch.tensor([float(it["row"]["effective_terminal_reward"]) for it in items])
    return dict(seq=seq.to(device), attn=attn.to(device), pw=pw, R=R, old=old, ref=ref, val=val, mask=mask, task_r=task_r)


@torch.no_grad()
def policy_logps(policy, T, device, mb):
    out = torch.zeros_like(T["old"])
    for s in range(0, T["seq"].shape[0], mb):
        sl = slice(s, s + mb)
        Rm = int(T["mask"][sl].sum(-1).max())
        seq, attn = T["seq"][sl][:, : T["pw"] + Rm], T["attn"][sl][:, : T["pw"] + Rm]
        lp, _ = token_logps(policy, seq, attn, T["pw"], seq[:, T["pw"]:])
        out[sl, :Rm] = lp.cpu()
    return out


def run_trajectory(policy, init_state, T, eps_train, epochs, lr, eps_list, mb, device):
    _load(policy, init_state)
    opt = AdamW(trainable_parameters(policy), lr=lr)
    scaler = torch.amp.GradScaler("cuda")
    total = float(T["mask"].sum())
    recs = []
    for step in range(epochs + 1):
        final = step == epochs
        opt.zero_grad(set_to_none=True)
        acc = {k: 0.0 for k in ["abs", "logr", "n"]}
        for e in eps_list:
            for k in ("aff", "blk", "clip_obj"):
                acc[f"{k}_{e}"] = 0.0
        acc["unclip_obj"], rmax, rmin = 0.0, -1e9, 1e9
        for s in range(0, T["seq"].shape[0], mb):
            sl = slice(s, s + mb)
            m_full = T["mask"][sl]
            Rm = int(m_full.sum(-1).max())
            seq, attn = T["seq"][sl][:, : T["pw"] + Rm], T["attn"][sl][:, : T["pw"] + Rm]
            old, adv, m = T["old"][sl][:, :Rm].to(device), T["adv"][sl][:, :Rm].to(device), m_full[:, :Rm].to(device)
            with torch.set_grad_enabled(not final):
                new_lp, _ = token_logps(policy, seq, attn, T["pw"], seq[:, T["pw"]:])
                if not final:
                    loss = ppo_policy_loss(new_lp, old, adv, m, eps_train)[0] * (m.sum() / total)
                    scaler.scale(loss).backward()
            with torch.no_grad():
                r = torch.exp(new_lp.detach() - old)
                mm = m.bool()
                acc["n"] += float(m.sum()); acc["abs"] += float(((r - 1).abs() * m).sum())
                acc["logr"] += float(((new_lp.detach() - old) * m).sum())
                rmax, rmin = max(rmax, float(r[mm].max())), min(rmin, float(r[mm].min()))
                acc["unclip_obj"] += float((r * adv * m).sum())
                for e in eps_list:
                    out = (r < 1 - e) | (r > 1 + e)
                    blocked = ((adv > 0) & (r > 1 + e)) | ((adv < 0) & (r < 1 - e))
                    acc[f"aff_{e}"] += float((out & mm).sum()); acc[f"blk_{e}"] += float((blocked & mm).sum())
                    acc[f"clip_obj_{e}"] += float((torch.minimum(r * adv, r.clamp(1 - e, 1 + e) * adv) * m).sum())
            del new_lp, r
        gn, skipped = float("nan"), False
        if not final:
            scaler.unscale_(opt)
            gn = float(torch.nn.utils.clip_grad_norm_(trainable_parameters(policy), 1.0))
            before = scaler.get_scale()
            scaler.step(opt); scaler.update()
            skipped = scaler.get_scale() < before
        n = acc["n"]
        rec = dict(step=step, grad_norm=gn, skipped_step=skipped, mean_abs_ratio_dev=acc["abs"] / n, mean_log_ratio=acc["logr"] / n,
                   ratio_max=rmax, ratio_min=rmin, surrogate_unclipped=acc["unclip_obj"] / n)
        for e in eps_list:
            rec[f"affected_frac_eps{e}"] = acc[f"aff_{e}"] / n
            rec[f"grad_blocked_frac_eps{e}"] = acc[f"blk_{e}"] / n
            rec[f"surrogate_clipped_eps{e}"] = acc[f"clip_obj_{e}"] / n
        recs.append(rec)
        print({k: (round(v, 5) if isinstance(v, float) else v) for k, v in rec.items() if k in ("step", "grad_norm", "skipped_step", "mean_abs_ratio_dev", "ratio_max", "ratio_min") or k.startswith("affected")}, flush=True)
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--mb", type=int, default=2)
    ap.add_argument("--tol", type=float, default=0.1, help="max mean |logp - cached old_logprobs| per row")
    ap.add_argument("--lr-mult", type=float, default=1.0, help="STRESS SETTING ONLY; default keeps the released LR")
    ap.add_argument("--name", default="cached_clipping")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    set_seed(int(cfg["seed"]))
    eps_list = [float(e) for e in cfg["clip_values"]]
    cache = load_cached_rollouts(cfg["cached_rollouts"])
    print("Cached PPO rollouts:", len(cache), "| eps values:", eps_list)

    tok = load_tokenizer(cfg["base_model"])
    pool = read_jsonl(cfg["paths"]["rl_prompt_train"]) + read_jsonl(cfg["paths"]["rl_prompt_eval"])  # cached rollouts come from the eval pool
    by_id = {r["prompt_id"]: r for r in pool}
    policy = load_policy(cfg, adapter_path=cfg["paths"]["ppo_midpoint_policy"], trainable=True)
    policy.eval()
    for p in trainable_parameters(policy):
        p.data = p.data.float()
    device = next(policy.parameters()).device

    # ---- 1. rebuild token ids from stored text, then check against the cached old_logprobs
    items, report = [], []
    for i, r in enumerate(cache):
        n = int(r["old_logprobs"].shape[0])
        prow = by_id.get(r.get("prompt_id"))
        if prow is None:
            report.append(dict(row=i, status="prompt_not_found")); continue
        rendered = tok.apply_chat_template(prompt_messages(prow), tokenize=False, add_generation_prompt=True)
        p_ids = tok(rendered, truncation=True, max_length=int(cfg["max_prompt_length"]))["input_ids"]
        ids = tok(r["response"], add_special_tokens=False)["input_ids"]
        if r.get("terminated_with_eos"):
            ids = ids + [tok.eos_token_id]
        status = "exact" if len(ids) == n else ("longer_truncated" if len(ids) > n else "shorter_dropped")
        report.append(dict(row=i, status=status, cached_len=n, retok_len=len(ids)))
        if len(ids) >= n:
            items.append(dict(i=i, p=p_ids, r=ids[:n], row=r))
    T0 = assemble(items, tok, device)
    new0 = policy_logps(policy, T0, device, args.mb)
    keep = []
    for b, it in enumerate(items):
        n = len(it["r"])
        d = float((new0[b, :n] - T0["old"][b, :n]).abs().mean())
        report[it["i"]]["mean_abs_logp_diff"] = d
        if d < args.tol:
            keep.append(it)
        else:
            report[it["i"]]["status"] += "+logp_mismatch_dropped"
    print(f"rows kept {len(keep)}/{len(cache)}; statuses:", pd.Series([r["status"] for r in report]).value_counts().to_dict())
    if len(keep) < 20:
        print("WARNING: fewer than 20 usable rows; do not report this study before checking the reconstruction")
    T = assemble(keep, tok, device)

    # ---- 2. advantages from the cached batch (same shaping/GAE/normalization as the training loop)
    beta, gamma, lam = float(cfg["kl_beta"]), float(cfg["gamma"]), float(cfg["gae_lambda"])
    rewards = shaped_rewards(T["task_r"], T["old"], T["ref"], T["mask"], beta)
    adv, ret = compute_gae(rewards, T["val"] * T["mask"], T["mask"], gamma, lam)
    T["adv"] = normalize_advantages(adv, T["mask"])
    print(f"batch: {len(keep)} rollouts, {int(T['mask'].sum())} response tokens | raw advantage mean {float(adv[T['mask'].bool()].mean()):.3f}")

    # ---- 3. trajectories from the identical starting weights
    lr = float(cfg["policy_learning_rate"]) * args.lr_mult
    init = _state(policy)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    results = {}
    for tag, e in [("unclipped", 1e9)] + [(f"eps_{x}", x) for x in eps_list]:
        print(f"\n=== trajectory {tag} (lr={lr:g}) ===", flush=True)
        results[tag] = run_trajectory(policy, init, T, e, args.epochs, lr, eps_list, args.mb, device)

    res_dir = repo_path(cfg["results_dir"]); res_dir.mkdir(parents=True, exist_ok=True)
    meta = dict(name=args.name, n_rollouts_cache=len(cache), n_rollouts_used=len(keep), n_tokens=int(T["mask"].sum()), eps_values=eps_list,
                epochs=args.epochs, lr=lr, lr_mult=args.lr_mult, kl_beta=beta, gamma=gamma, gae_lambda=lam, seed=int(cfg["seed"]),
                logp_tolerance=args.tol, microbatch=args.mb, wall_clock_s=time.time() - t0,
                peak_vram_gb=torch.cuda.max_memory_allocated() / 2**30, source_indices_used=[it["row"]["source_index"] for it in keep])
    (res_dir / f"{args.name}.json").write_text(json.dumps(dict(meta=meta, reconstruction=report, trajectories=results), indent=2, default=float))

    print("\n=== affected-token fraction on the UNCLIPPED trajectory (same ratios, three eps) ===")
    df = pd.DataFrame(results["unclipped"])
    cols = ["step", "mean_abs_ratio_dev", "ratio_max", "ratio_min"] + [f"affected_frac_eps{e}" for e in eps_list] + [f"grad_blocked_frac_eps{e}" for e in eps_list]
    print(df[cols].round(4).to_string(index=False))
    print("\n=== each eps trained with its own clipped objective: affected fraction at its own eps, last step ===")
    for e in eps_list:
        d = pd.DataFrame(results[f"eps_{e}"])
        print(f"eps={e}: step0 {d[f'affected_frac_eps{e}'].iloc[0]:.4f} -> last {d[f'affected_frac_eps{e}'].iloc[-1]:.4f} | max ratio dev last {d['mean_abs_ratio_dev'].iloc[-1]:.4f}")
    print("meta:", {k: meta[k] for k in ("n_rollouts_used", "n_tokens", "lr", "wall_clock_s", "peak_vram_gb")})


if __name__ == "__main__":
    main()

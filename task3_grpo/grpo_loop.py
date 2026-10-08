from __future__ import annotations

import json
import shutil
import time

import numpy as np
import torch

from common.data import prompt_messages, repo_path
from common.generation import batch_generate, score_reward_pairs
from common.logging_utils import set_seed
from common.models import reference_mode, trainable_parameters
from task2_ppo.ppo_loop import _load, _state, token_logps
from task3_grpo.grpo import group_relative_advantages, grpo_policy_loss, mask_truncated_sequences


@torch.no_grad()
def sample_group(policy, tok, rm, rm_tok, cfg, messages, K):
    gen = cfg["generation"]
    policy.config.use_cache = True
    try:
        g = batch_generate(policy, tok, [messages] * K, max_prompt_length=int(cfg["max_prompt_length"]),
                           max_new_tokens=int(cfg["max_completion_length"]), temperature=float(gen["temperature"]),
                           top_p=float(gen["top_p"]), do_sample=bool(gen["do_sample"]))
    finally:
        policy.config.use_cache = False
    seq, attn, resp, rmask = (g[k].clone() for k in ("sequences", "attention_mask", "response_ids", "response_mask"))
    raw = score_reward_pairs(rm, rm_tok, [messages] * K, g["responses"],
                             max_length=int(cfg.get("reward_max_length", 1280))).float().cpu()
    return dict(seq=seq, attn=attn, resp=resp, rmask=rmask, pw=g["prompt_width"], rewards=raw,
                truncated=[bool(t) for t in g["truncated"]], lengths=[int(n) for n in g["response_lengths"]])


def update_once(policy, opt, scaler, tok, rm, rm_tok, cfg, rows, idx, loss_type, dev):
    K, eps, beta = int(cfg["num_generations"]), float(cfg["clip_epsilon"]), float(cfg["kl_beta"])
    maxc, ppu = int(cfg["max_completion_length"]), len(idx)
    opt.zero_grad(set_to_none=True)
    a = {k: [] for k in ("reward", "std", "kl", "ent", "clip", "pol", "lens", "rews", "advs", "trunc", "w")}
    for i in idx:
        S = sample_group(policy, tok, rm, rm_tok, cfg, prompt_messages(rows[i]), K)
        adv = group_relative_advantages(S["rewards"], torch.zeros(K, dtype=torch.long))
        mask = S["rmask"]
        if cfg.get("mask_truncated_completions", True):
            mask = mask_truncated_sequences(mask, S["truncated"])
        seq, attn, resp, pw = S["seq"], S["attn"], S["resp"], S["pw"]
        with torch.no_grad():
            old, _ = token_logps(policy, seq, attn, pw, resp)
            with reference_mode(policy):
                ref, _ = token_logps(policy, seq, attn, pw, resp)
        new, _ = token_logps(policy, seq, attn, pw, resp)
        loss, d = grpo_policy_loss(new, old, adv.to(dev), mask.to(dev), ref, eps, beta, loss_type, maxc)
        scaler.scale(loss / ppu).backward()
        r, T = S["rewards"], np.array(S["lengths"], dtype=float)
        a["reward"].append(float(r.mean())); a["std"].append(float(r.std(unbiased=False)))
        a["kl"].append(float(d["sampled_kl"])); a["ent"].append(float(d["sample_entropy"]))
        a["clip"].append(float(d["clip_fraction"])); a["pol"].append(float(d["policy_term"]))
        a["lens"] += S["lengths"]; a["rews"] += r.tolist(); a["advs"] += adv.tolist(); a["trunc"] += S["truncated"]
        a["w"] += (adv.abs().numpy() / (T.clip(min=1) if loss_type == "grpo" else maxc)).tolist()
        del new, old, ref, loss, S
    scaler.unscale_(opt)
    gn = float(torch.nn.utils.clip_grad_norm_(trainable_parameters(policy), float(cfg["max_grad_norm"])))
    before = scaler.get_scale(); scaler.step(opt); scaler.update()
    return dict(a, grad_norm=gn, skipped=bool(scaler.get_scale() < before), scale=scaler.get_scale())


def optimize(bundle, cfg, out, run_name, loss_type="grpo", save_every=5):
    tok, policy, rm, rm_tok = bundle["tokenizer"], bundle["policy"], bundle["reward_model"], bundle["reward_tokenizer"]
    opt, rows = bundle["optimizer"], bundle["prompt_rows"]
    n_up, ppu, seed = int(cfg["updates"]), int(cfg["prompts_per_update"]), int(cfg["seed"])
    sel = np.random.RandomState(seed).permutation(len(rows))[: n_up * ppu]  # same seeded order as the PPO runs
    for p in trainable_parameters(policy):
        if p.dtype != torch.float32:
            p.data = p.data.float()
    for m in policy.modules():            # no LoRA dropout, so old == new logprobs on the first pass
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0
    policy.train()
    dev = next(policy.parameters()).device
    scaler = torch.amp.GradScaler("cuda")
    res = repo_path(cfg["results_dir"]); res.mkdir(parents=True, exist_ok=True); out.mkdir(parents=True, exist_ok=True)
    log, ckpt = res / f"train_{run_name}.jsonl", out / "ckpt.pt"
    start, prev = 0, 0.0
    if ckpt.exists():
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        _load(policy, ck["policy"]); opt.load_state_dict(ck["opt"]); scaler.load_state_dict(ck["scaler"])
        start, prev = int(ck["update"]), float(ck["elapsed"])
        if not log.exists() and (out / log.name).exists():
            shutil.copy(out / log.name, log)
        print(f"RESUMING from update {start}/{n_up}")
    elif log.exists():
        log.unlink()
    torch.cuda.reset_peak_memory_stats(); t_start = time.time()
    for u in range(start, n_up):
        t0 = time.time(); set_seed(seed + u)
        idx = [int(i) for i in sel[u * ppu:(u + 1) * ppu]]
        s = update_once(policy, opt, scaler, tok, rm, rm_tok, cfg, rows, idx, loss_type, dev)
        row = dict(update=u + 1, run=run_name, loss_type=loss_type, prompt_index=idx, reward=float(np.mean(s["reward"])),
                   group_reward_std=float(np.mean(s["std"])), uninformative_group_frac=float(np.mean([x <= 1e-6 for x in s["std"]])),
                   kl=float(np.mean(s["kl"])), entropy=float(np.mean(s["ent"])), clip_fraction=float(np.mean(s["clip"])),
                   policy_term=float(np.mean(s["pol"])), grad_norm=s["grad_norm"], step_skipped=s["skipped"], loss_scale=s["scale"],
                   response_len=float(np.mean(s["lens"])), n_truncated=int(sum(s["trunc"])), lens=s["lens"], rewards=s["rews"],
                   advantages=s["advs"], truncated=s["trunc"], per_token_weight=s["w"], update_s=time.time() - t0,
                   elapsed_s=prev + time.time() - t_start, peak_vram_gb=torch.cuda.max_memory_allocated() / 2**30)
        with log.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()
               if k not in ("lens", "rewards", "advantages", "truncated", "per_token_weight", "run", "prompt_index")}, flush=True)
        torch.cuda.empty_cache()
        if (u + 1) % save_every == 0 or (u + 1) == n_up:
            policy.save_pretrained(str(out))
            torch.save(dict(policy=_state(policy), opt=opt.state_dict(), scaler=scaler.state_dict(), update=u + 1,
                            elapsed=prev + time.time() - t_start), ckpt)
            shutil.copy(log, out / log.name)
    summary = dict(run_name=run_name, loss_type=loss_type, updates=n_up, num_generations=int(cfg["num_generations"]),
                   prompts_per_update=ppu, seed=seed, learning_rate=float(cfg["learning_rate"]), prompt_indices=[int(i) for i in sel],
                   wall_clock_s=prev + time.time() - t_start, peak_vram_gb_this_session=torch.cuda.max_memory_allocated() / 2**30,
                   gpu=torch.cuda.get_device_name(0), midpoint_policy=cfg["paths"]["grpo_midpoint_policy"], output=str(out),
                   note="LoRA dropout set to 0; reference = base model (adapter disabled); truncated completions masked")
    (res / f"train_{run_name}_summary.json").write_text(json.dumps(summary, indent=2))
    return summary

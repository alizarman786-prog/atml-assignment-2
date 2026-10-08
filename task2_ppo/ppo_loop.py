from __future__ import annotations

import json
import shutil
import time

import numpy as np
import torch

from common.data import prompt_messages, repo_path
from common.generation import batch_generate, score_reward_pairs
from common.logging_utils import set_seed
from common.metrics import masked_mean, sampled_kl
from common.models import reference_mode, token_values, trainable_parameters
from task2_ppo.ppo import compute_gae, normalize_advantages, ppo_policy_loss, shaped_rewards, value_mse_loss


def token_logps(model, seq, attn, pw, resp, entropy=False):
    pos = (attn.cumsum(-1) - 1).clamp(min=0)
    logits = model(input_ids=seq, attention_mask=attn, position_ids=pos, use_cache=False).logits
    logits = logits[:, pw - 1:-1, :][:, : resp.shape[1], :].float()
    logp_all = torch.log_softmax(logits, dim=-1)
    logp = torch.gather(logp_all, -1, resp.unsqueeze(-1)).squeeze(-1)
    ent = None
    if entropy:
        with torch.no_grad():
            ent = -(logp_all.exp() * logp_all).sum(-1)
    return logp, ent


def response_values(value_model, seq, attn, pw, n):
    with torch.autocast("cuda", dtype=torch.float16):
        v = token_values(value_model, seq, attn)
    return v.float()[:, pw - 1: pw - 1 + n]


def _state(m):
    return {k: p.detach().cpu() for k, p in m.named_parameters() if p.requires_grad}


def _load(m, sd):
    params = dict(m.named_parameters())
    with torch.no_grad():
        for k, v in sd.items():
            params[k].copy_(v.to(params[k].device, params[k].dtype))


def optimize(bundle, cfg, out, run_name, save_every=5):
    tok, policy, value = bundle["tokenizer"], bundle["policy"], bundle["value_model"]
    rm, rm_tok = bundle["reward_model"], bundle["reward_tokenizer"]
    popt, vopt = bundle["policy_optimizer"], bundle["value_optimizer"]
    rows = bundle["prompt_rows"]
    n_up, ppu, epochs = int(cfg["updates"]), int(cfg["prompts_per_update"]), int(cfg["ppo_epochs"])
    eps, beta = float(cfg["clip_epsilon"]), float(cfg["kl_beta"])
    gamma, lam, vcoef = float(cfg["gamma"]), float(cfg["gae_lambda"]), float(cfg["value_coef"])
    clip_norm, eos_pen, seed = float(cfg["max_grad_norm"]), float(cfg.get("missing_eos_penalty", 0.0)), int(cfg["seed"])
    gen = cfg["generation"]

    # fixed prompt order: seeded permutation; short forks use the identical prefix
    sel = np.random.RandomState(seed).permutation(len(rows))[: n_up * ppu]

    # LoRA / head parameters in fp32 so GradScaler can unscale; frozen weights stay fp16
    for m in (policy, value):
        for p in trainable_parameters(m):
            if p.dtype != torch.float32:
                p.data = p.data.float()
    # eval mode: no LoRA dropout, so rho == 1 at the first PPO epoch and clip fraction measures real movement
    policy.eval(); value.eval()
    pscaler, vscaler = torch.amp.GradScaler("cuda"), torch.amp.GradScaler("cuda")

    res_dir = repo_path(cfg["results_dir"]); res_dir.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    log_path, ckpt_path = res_dir / f"train_{run_name}.jsonl", out / "ckpt.pt"
    start, elapsed_prev = 0, 0.0
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        _load(policy, ck["policy"]); _load(value, ck["value"])
        popt.load_state_dict(ck["popt"]); vopt.load_state_dict(ck["vopt"])
        pscaler.load_state_dict(ck["pscaler"]); vscaler.load_state_dict(ck["vscaler"])
        start, elapsed_prev = int(ck["update"]), float(ck["elapsed"])
        if not log_path.exists() and (out / log_path.name).exists():
            shutil.copy(out / log_path.name, log_path)
        print(f"RESUMING from update {start}/{n_up}")
    elif log_path.exists():
        log_path.unlink()

    torch.cuda.reset_peak_memory_stats()
    t_start = time.time()
    for u in range(start, n_up):
        t0 = time.time()
        set_seed(seed + u)
        idx = [int(i) for i in sel[u * ppu:(u + 1) * ppu]]
        prompts = [prompt_messages(rows[i]) for i in idx]

        policy.config.use_cache = True
        try:
            g = batch_generate(policy, tok, prompts, max_prompt_length=int(cfg["max_prompt_length"]),
                               max_new_tokens=int(cfg["max_response_length"]), temperature=float(gen["temperature"]),
                               top_p=float(gen["top_p"]), do_sample=bool(gen["do_sample"]))
        finally:
            policy.config.use_cache = False
        seq, attn, resp, rmask, pw = g["sequences"], g["attention_mask"], g["response_ids"], g["response_mask"], g["prompt_width"]
        R = resp.shape[1]

        with torch.no_grad():
            old_lp, _ = token_logps(policy, seq, attn, pw, resp)
            with reference_mode(policy):
                ref_lp, _ = token_logps(policy, seq, attn, pw, resp)
            old_v = response_values(value, seq, attn, pw, R) * rmask
            raw = score_reward_pairs(rm, rm_tok, prompts, g["responses"], max_length=int(cfg["reward_max_length"])).float().cpu()
            pen = torch.tensor([0.0 if t else eos_pen for t in g["terminated_with_eos"]])
            task_r = (raw - pen).to(seq.device)
            rewards = shaped_rewards(task_r, old_lp, ref_lp, rmask, beta)
            adv, ret = compute_gae(rewards, old_v, rmask, gamma, lam)
            adv_n = normalize_advantages(adv, rmask)
            m = rmask.bool()
            kl_tok = float(sampled_kl(old_lp, ref_lp, rmask))
            kl_seq = float(((old_lp - ref_lp) * rmask).sum(-1).mean())
            rv, vv = ret[m], old_v[m]
            ev = float(1 - (rv - vv).var() / rv.var().clamp_min(1e-8)) if rv.numel() > 1 else float("nan")
            vcorr = float(torch.corrcoef(torch.stack([rv, vv]))[0, 1]) if rv.numel() > 1 else float("nan")

        pl, vl, pgn, vgn, cf = [], [], [], [], []
        ent0 = None
        for e in range(epochs):
            new_lp, ent = token_logps(policy, seq, attn, pw, resp, entropy=(e == 0))
            if e == 0:
                ent0 = float(masked_mean(ent, rmask))
            ploss, _, clipf = ppo_policy_loss(new_lp, old_lp, adv_n, rmask, eps)
            popt.zero_grad(set_to_none=True)
            pscaler.scale(ploss).backward()
            pscaler.unscale_(popt)
            pgn.append(float(torch.nn.utils.clip_grad_norm_(trainable_parameters(policy), clip_norm)))
            pscaler.step(popt); pscaler.update()
            pl.append(float(ploss.detach())); cf.append(float(clipf))
            del new_lp, ent, ploss

            new_v = response_values(value, seq, attn, pw, R)
            vloss = value_mse_loss(new_v, ret, rmask)
            vopt.zero_grad(set_to_none=True)
            vscaler.scale(vcoef * vloss).backward()
            vscaler.unscale_(vopt)
            vgn.append(float(torch.nn.utils.clip_grad_norm_(trainable_parameters(value), clip_norm)))
            vscaler.step(vopt); vscaler.update()
            vl.append(float(vloss.detach()))
            del new_v, vloss

        term = [bool(t) for t in g["terminated_with_eos"]]
        row = dict(update=u + 1, run=run_name, prompt_index=idx, prompt_ids=[rows[i].get("prompt_id", i) for i in idx],
                   task_reward_raw=float(raw.mean()), task_reward=float(task_r.mean()), missing_eos=1.0 - float(np.mean(term)),
                   kl_token=kl_tok, kl_seq_sum=kl_seq, entropy=ent0, policy_loss=float(np.mean(pl)), value_loss=float(np.mean(vl)),
                   value_loss_first_epoch=vl[0], grad_norm_policy=float(np.mean(pgn)), grad_norm_value=float(np.mean(vgn)),
                   clip_fraction=float(np.mean(cf)), clip_fraction_last_epoch=cf[-1],
                   response_len=float(rmask.sum(-1).mean()), value_mean=float(old_v[m].mean()), return_mean=float(ret[m].mean()),
                   value_corr=vcorr, value_explained_var=ev, kl_beta=beta, clip_eps=eps,
                   update_s=time.time() - t0, elapsed_s=elapsed_prev + time.time() - t_start,
                   peak_vram_gb=torch.cuda.max_memory_allocated() / 2**30)
        with log_path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items() if k not in ("prompt_ids", "run")}, flush=True)

        del g, seq, attn, resp, rmask, old_lp, ref_lp, old_v, rewards, adv, ret, adv_n
        torch.cuda.empty_cache()

        if (u + 1) % save_every == 0 or (u + 1) == n_up:
            policy.save_pretrained(str(out))
            torch.save({"policy": _state(policy), "value": _state(value), "popt": popt.state_dict(), "vopt": vopt.state_dict(),
                        "pscaler": pscaler.state_dict(), "vscaler": vscaler.state_dict(), "update": u + 1,
                        "elapsed": elapsed_prev + time.time() - t_start}, ckpt_path)
            shutil.copy(log_path, out / log_path.name)

    summary = dict(run_name=run_name, updates=n_up, clip_epsilon=eps, kl_beta=beta, ppo_epochs=epochs, prompts_per_update=ppu,
                   seed=seed, policy_lr=float(cfg["policy_learning_rate"]), prompt_indices=[int(i) for i in sel],
                   wall_clock_s=elapsed_prev + time.time() - t_start, peak_vram_gb_this_session=torch.cuda.max_memory_allocated() / 2**30,
                   gpu=torch.cuda.get_device_name(0), midpoint_policy=cfg["paths"]["ppo_midpoint_policy"], output=str(out),
                   note="policy and critic in eval mode (no LoRA dropout) during updates; reference = base model (adapter disabled)")
    (res_dir / f"train_{run_name}_summary.json").write_text(json.dumps(summary, indent=2))
    return summary

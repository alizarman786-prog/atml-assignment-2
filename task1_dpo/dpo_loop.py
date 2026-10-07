from __future__ import annotations

import json
import time

import torch
import torch.nn.functional as F
from peft import set_peft_model_state_dict
from safetensors.torch import load_file
from torch.utils.data import DataLoader

from common.data import repo_path
from common.models import reference_mode, trainable_parameters
from task1_dpo.dpo import dpo_loss


def sequence_logps(model, batch, device):
    """Sum of response-token log-probs per sequence (prompt/padding masked out)."""
    input_ids = batch["input_ids"].to(device)
    attn = batch["attention_mask"].to(device)
    rmask = batch["response_mask"].to(device)
    pos = (attn.cumsum(-1) - 1).clamp(min=0)
    logits = model(input_ids=input_ids, attention_mask=attn, position_ids=pos, use_cache=False).logits[:, :-1]
    labels = input_ids[:, 1:]
    mask = rmask[:, 1:].bool()
    tok = torch.zeros(mask.shape, device=device, dtype=torch.float32)
    tok[mask] = -F.cross_entropy(logits[mask].float(), labels[mask], reduction="none")
    return tok.sum(-1), mask.sum(-1)


def optimize(bundle, output, run_name, dataset_path, max_examples=None, save_every=20):
    cfg, model, opt, beta = bundle["cfg"], bundle["model"], bundle["optimizer"], bundle["beta"]
    rows = bundle["rows"]
    collate = bundle["loader"].collate_fn
    device = next(model.parameters()).device
    bs, accum = int(cfg["batch_size"]), int(cfg["grad_accum_steps"])
    epochs, clip, seed = int(cfg["epochs"]), float(cfg["max_grad_norm"]), int(cfg["seed"])

    # GradScaler cannot unscale fp16 gradients, so keep trainable (LoRA) params in fp32.
    for p in trainable_parameters(model):
        if p.dtype == torch.float16:
            p.data = p.data.float()
    print("trainable dtypes:", {str(p.dtype) for p in trainable_parameters(model)})

    n_mb = (len(rows) + bs - 1) // bs
    steps_per_epoch = (n_mb + accum - 1) // accum
    total_steps = steps_per_epoch * epochs

    output.mkdir(parents=True, exist_ok=True)
    results_dir = repo_path(cfg["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    log_path = results_dir / f"train_{run_name}.jsonl"
    state_path = output / "trainer_state.pt"
    scaler = torch.amp.GradScaler("cuda")

    start_step = 0
    if state_path.exists():
        st = torch.load(state_path, map_location="cpu", weights_only=False)
        set_peft_model_state_dict(model, load_file(str(output / "adapter_model.safetensors")))
        opt.load_state_dict(st["opt"])
        scaler.load_state_dict(st["scaler"])
        start_step = int(st["step"])
        print(f"RESUMING {output} from step {start_step}/{total_steps}")
    elif log_path.exists():
        log_path.unlink()

    done_epochs, r = divmod(start_step, steps_per_epoch)
    skip = r * accum
    model.train()
    torch.cuda.reset_peak_memory_stats()
    t0, step = time.time(), start_step
    s = dict(loss=0.0, acc=0.0, margin=0.0, kl=0.0)

    def save():
        model.save_pretrained(str(output))
        torch.save({"opt": opt.state_dict(), "scaler": scaler.state_dict(), "step": step}, state_path)

    for epoch in range(done_epochs, epochs):
        g = torch.Generator().manual_seed(seed + epoch)
        loader = DataLoader(rows, batch_size=bs, shuffle=True, collate_fn=collate, generator=g)
        for i, (chosen, rejected) in enumerate(loader):
            if epoch == done_epochs and i < skip:
                continue
            gsize = min(accum, n_mb - (i // accum) * accum)

            with torch.no_grad(), reference_mode(model):
                rc, _ = sequence_logps(model, chosen, device)
                rr, _ = sequence_logps(model, rejected, device)
            pc, nc = sequence_logps(model, chosen, device)
            pr, _ = sequence_logps(model, rejected, device)

            loss, d = dpo_loss(pc, pr, rc, rr, beta)
            scaler.scale(loss / gsize).backward()

            with torch.no_grad():
                s["loss"] += loss.item() / gsize
                s["acc"] += d["preference_accuracy"].item() / gsize
                s["margin"] += (((pc - rc) - (pr - rr)).mean().item()) / gsize
                s["kl"] += ((pc - rc).sum() / nc.sum()).item() / gsize  # chosen-token log-ratio proxy

            if (i + 1) % accum == 0 or i == n_mb - 1:
                scaler.unscale_(opt)
                gn = torch.nn.utils.clip_grad_norm_(trainable_parameters(model), clip).item()
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                step += 1
                row = dict(step=step, epoch=epoch, beta=beta, loss=s["loss"], pref_acc=s["acc"],
                           margin=s["margin"], chosen_logratio_per_token=s["kl"], grad_norm=gn,
                           loss_scale=scaler.get_scale(), elapsed_s=time.time() - t0,
                           peak_vram_gb=torch.cuda.max_memory_allocated() / 2**30)
                with log_path.open("a") as f:
                    f.write(json.dumps(row) + "\n")
                print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})
                s = dict(loss=0.0, acc=0.0, margin=0.0, kl=0.0)
                if step % save_every == 0 or step == total_steps:
                    save()

    save()
    summary = dict(run_name=run_name, beta=beta, dataset=dataset_path, n_examples=len(rows),
                   max_examples=max_examples, steps=step, total_steps=total_steps, batch_size=bs,
                   grad_accum_steps=accum, learning_rate=float(cfg["learning_rate"]), seed=seed,
                   epochs=epochs, wall_clock_s_this_session=time.time() - t0,
                   peak_vram_gb=torch.cuda.max_memory_allocated() / 2**30, output=str(output))
    (results_dir / f"train_{run_name}_summary.json").write_text(json.dumps(summary, indent=2))
    return summary

from __future__ import annotations
import json, math, os, random, time
import torch
import torch.nn.functional as F


def _pad_id(tokenizer):
    for name in ("pad_token_id", "eos_token_id"):
        v = getattr(tokenizer, name, None)
        if v is not None:
            return v
    return 0


def build_sft_batch(tokenizer, prompts, responses, max_length):
    eos = getattr(tokenizer, "eos_token_id", None)
    seqs, labs = [], []
    for p, r in zip(prompts, responses):
        p_ids = tokenizer.encode(p, add_special_tokens=False)
        r_ids = tokenizer.encode(r, add_special_tokens=False)
        if eos is not None:
            r_ids = r_ids + [eos]
        seqs.append((p_ids + r_ids)[:max_length])
        labs.append(([-100] * len(p_ids) + r_ids)[:max_length])
    width = max(len(s) for s in seqs) if seqs else 0
    pad = _pad_id(tokenizer)
    input_ids = torch.full((len(seqs), width), pad, dtype=torch.long)
    labels = torch.full((len(seqs), width), -100, dtype=torch.long)
    attention_mask = torch.zeros((len(seqs), width), dtype=torch.long)
    for i, (s, l) in enumerate(zip(seqs, labs)):
        input_ids[i, :len(s)] = torch.tensor(s, dtype=torch.long)
        labels[i, :len(l)] = torch.tensor(l, dtype=torch.long)
        attention_mask[i, :len(s)] = 1
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def logps_from_batch(model, batch):
    dev = next(model.parameters()).device
    ids = batch["input_ids"].to(dev)
    am = batch["attention_mask"].to(dev)
    labels = batch["labels"].to(dev)
    logits = model(input_ids=ids, attention_mask=am).logits[:, :-1, :].float()
    tgt = labels[:, 1:]
    B, T, V = logits.shape
    nll = F.cross_entropy(logits.reshape(-1, V), tgt.reshape(-1),
                          ignore_index=-100, reduction="none")
    return -nll.view(B, T).sum(dim=1)


def compute_response_logprobs(model, tokenizer, prompts, responses, max_length):
    return logps_from_batch(model, build_sft_batch(tokenizer, prompts, responses, max_length))


def sft_loss_from_batch(model, batch):
    logps = logps_from_batch(model, batch)
    n = batch["labels"][:, 1:].ne(-100).sum().clamp_min(1).to(logps.device)
    return -logps.sum() / n  # token-mean reduction


def compute_sft_loss(model, tokenizer, prompts, responses, max_length):
    return sft_loss_from_batch(model, build_sft_batch(tokenizer, prompts, responses, max_length))


def _lr_lambda(total, warmup_frac=0.03):
    w = max(1, int(total * warmup_frac))

    def f(s):
        if s < w:
            return (s + 1) / w
        return 0.5 * (1 + math.cos(math.pi * (s - w) / max(1, total - w)))
    return f


@torch.no_grad()
def evaluate_sft(model, tok, val_rows, max_length, bs=4):
    model.eval()
    nll, ntok = 0.0, 0
    for i in range(0, len(val_rows), bs):
        rows = val_rows[i:i + bs]
        b = build_sft_batch(tok, [r["prompt"] for r in rows],
                            [r["response"] for r in rows], max_length)
        with torch.autocast("cuda", dtype=torch.float16):
            lp = logps_from_batch(model, b)
        nll += -lp.sum().item()
        ntok += b["labels"][:, 1:].ne(-100).sum().item()
    model.train()
    return nll / max(1, ntok)


def train_sft(train_rows, val_rows, out_dir, model_name="Qwen/Qwen2.5-0.5B", max_length=512,
              micro_bs=2, grad_accum=16, lr=2e-5, weight_decay=0.1, warmup_frac=0.03,
              clip=1.0, epochs=1, eval_every=100, log_every=10, seed=42, device="cuda:0"):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    os.makedirs(out_dir, exist_ok=True)
    random.seed(seed)
    torch.manual_seed(seed)
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32).to(device)
    model.config.use_cache = False
    model.train()

    decay, nodecay = [], []
    for n, p in model.named_parameters():
        (nodecay if (p.ndim < 2 or "norm" in n) else decay).append(p)
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": weight_decay},
         {"params": nodecay, "weight_decay": 0.0}], lr=lr, betas=(0.9, 0.95))

    micro_per_epoch = len(train_rows) // micro_bs
    total = (micro_per_epoch // grad_accum) * epochs
    sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda(total, warmup_frac))
    scaler = torch.amp.GradScaler("cuda")
    print("optimizer steps:", total)

    mpath = os.path.join(out_dir, "metrics.jsonl")

    def log(d):
        with open(mpath, "a") as f:
            f.write(json.dumps(d) + "\n")

    def save(tag):
        p = os.path.join(out_dir, tag)
        sd = {k: v.detach().to(torch.float16) for k, v in model.state_dict().items()}
        model.save_pretrained(p, state_dict=sd, safe_serialization=True)
        tok.save_pretrained(p)

    best, step, t0, rl, rn = float("inf"), 0, time.time(), 0.0, 0
    for ep in range(epochs):
        order = list(range(len(train_rows)))
        random.Random(seed + ep).shuffle(order)
        for mb in range(micro_per_epoch):
            rows = [train_rows[i] for i in order[mb * micro_bs:(mb + 1) * micro_bs]]
            b = build_sft_batch(tok, [r["prompt"] for r in rows],
                                [r["response"] for r in rows], max_length)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = sft_loss_from_batch(model, b)
            scaler.scale(loss / grad_accum).backward()
            rl += loss.item()
            rn += 1
            if (mb + 1) % grad_accum == 0:
                scaler.unscale_(opt)
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
                step += 1
                if step % log_every == 0:
                    r = {"step": step, "train_loss": rl / rn, "lr": sched.get_last_lr()[0],
                         "grad_norm": float(gn), "elapsed_s": time.time() - t0}
                    log(r)
                    print(r)
                    rl, rn = 0.0, 0
                if step % eval_every == 0 or step == total:
                    v = evaluate_sft(model, tok, val_rows, max_length)
                    log({"step": step, "val_loss": v})
                    print(f"[val] step {step} {v:.4f}")
                    if v < best:
                        best = v
                        save("best")
                        print("  saved best")
    save("final")
    log({"done": True, "best_val_loss": best})
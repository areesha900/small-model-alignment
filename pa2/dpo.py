from __future__ import annotations
import torch
import torch.nn.functional as F
from pa2.sft import build_sft_batch, logps_from_batch
import json, os, random, time


def build_dpo_batch(tokenizer, prompts, chosen_responses, rejected_responses, max_length):
    return {"chosen": build_sft_batch(tokenizer, prompts, chosen_responses, max_length),
            "rejected": build_sft_batch(tokenizer, prompts, rejected_responses, max_length)}


def dpo_loss_from_logps(pc, pr, rc, rr, beta):
    rc, rr = rc.detach().to(pc.device), rr.detach().to(pc.device)
    return -F.logsigmoid(beta * ((pc - pr) - (rc - rr))).mean()


def dpo_metrics_from_logps(pc, pr, rc, rr, beta):
    rc, rr = rc.detach().to(pc.device), rr.detach().to(pc.device)
    rw, rl = beta * (pc - rc), beta * (pr - rr)
    return {"reward_accuracy": (rw > rl).float().mean().item(),
            "reward_margin": (rw - rl).mean().item()}


def policy_and_reference_logps(policy, reference, batch):
    pc = logps_from_batch(policy, batch["chosen"])
    pr = logps_from_batch(policy, batch["rejected"])
    with torch.no_grad():
        rc = logps_from_batch(reference, batch["chosen"])
        rr = logps_from_batch(reference, batch["rejected"])
    return pc, pr, rc, rr


def compute_dpo_loss(policy_model, reference_model, tokenizer, prompts, chosen_responses,
                     rejected_responses, beta, max_length):
    b = build_dpo_batch(tokenizer, prompts, chosen_responses, rejected_responses, max_length)
    return dpo_loss_from_logps(*policy_and_reference_logps(policy_model, reference_model, b), beta)

# --------------------------------------------------------------------------------------
# Training loop (from scratch: no HF Trainer / TRL)
# --------------------------------------------------------------------------------------

def filter_long_prompts(rows, tokenizer, max_prompt_tokens=256):
    """Drop pairs whose prompt exceeds max_prompt_tokens (so responses are never squeezed out)."""
    return [r for r in rows
            if len(tokenizer.encode(r["prompt"], add_special_tokens=False)) <= max_prompt_tokens
            and r["chosen"] and r["rejected"]]


def _save_fp16(model, tok, path):
    sd = {k: v.detach().to(torch.float16) for k, v in model.state_dict().items()}
    model.save_pretrained(path, state_dict=sd, safe_serialization=True)
    tok.save_pretrained(path)


@torch.no_grad()
def evaluate_dpo(policy, reference, tok, val_rows, beta, max_length, ref_cache=None):
    """Mean DPO loss, reward accuracy (strict rw > rl) and reward margin on validation pairs.
    The reference is frozen, so its log-probs are computed once and cached in `ref_cache`."""
    policy.eval()
    losses, accs, margins = [], [], []
    for i, r in enumerate(val_rows):
        b = build_dpo_batch(tok, [r["prompt"]], [r["chosen"]], [r["rejected"]], max_length)
        with torch.autocast("cuda", dtype=torch.float16):
            pc = logps_from_batch(policy, b["chosen"])
            pr = logps_from_batch(policy, b["rejected"])
            if ref_cache is not None and i in ref_cache:
                rc, rr = ref_cache[i]
            else:
                rc = logps_from_batch(reference, b["chosen"])
                rr = logps_from_batch(reference, b["rejected"])
                if ref_cache is not None:
                    ref_cache[i] = (rc.cpu(), rr.cpu())
        losses.append(dpo_loss_from_logps(pc, pr, rc, rr, beta).item())
        m = dpo_metrics_from_logps(pc, pr, rc, rr, beta)
        accs.append(m["reward_accuracy"])
        margins.append(m["reward_margin"])
    policy.train()
    n = max(1, len(val_rows))
    return {"val_loss": sum(losses) / n, "val_reward_acc": sum(accs) / n,
            "val_reward_margin": sum(margins) / n}


def train_dpo(train_rows, val_rows, out_dir, sft_path, beta=0.1, max_length=512, grad_accum=32,
              lr=1e-6, weight_decay=0.0, clip=1.0, epochs=1, eval_every=100, log_every=10,
              seed=42, policy_device="cuda:0", ref_device=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    os.makedirs(out_dir, exist_ok=True)
    random.seed(seed)
    torch.manual_seed(seed)
    if ref_device is None:
        ref_device = "cuda:1" if torch.cuda.device_count() > 1 else policy_device
    tok = AutoTokenizer.from_pretrained(sft_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    policy = AutoModelForCausalLM.from_pretrained(sft_path, torch_dtype=torch.float32).to(policy_device)
    policy.config.use_cache = False
    policy.train()
    reference = AutoModelForCausalLM.from_pretrained(sft_path, torch_dtype=torch.float16).to(ref_device)
    reference.config.use_cache = False
    reference.eval()
    for p in reference.parameters():
        p.requires_grad_(False)

    opt = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler("cuda")
    total = (len(train_rows) // grad_accum) * epochs
    print("optimizer steps:", total)

    mpath = os.path.join(out_dir, "metrics.jsonl")

    def log(d):
        with open(mpath, "a") as f:
            f.write(json.dumps(d) + "\n")

    ref_cache = {}
    best_acc, best_margin, step = -1.0, float("-inf"), 0

    def run_eval(tag_step):
        nonlocal best_acc, best_margin
        m = evaluate_dpo(policy, reference, tok, val_rows, beta, max_length, ref_cache)
        m["step"] = tag_step
        log(m)
        print("[val]", m)
        if (m["val_reward_acc"], m["val_reward_margin"]) > (best_acc, best_margin):
            best_acc, best_margin = m["val_reward_acc"], m["val_reward_margin"]
            _save_fp16(policy, tok, os.path.join(out_dir, "best"))
            print("  saved best")

    run_eval(0) 
    t0, acc_l, acc_a, acc_m, rn = time.time(), 0.0, 0.0, 0.0, 0
    for ep in range(epochs):
        order = list(range(len(train_rows)))
        random.Random(seed + ep).shuffle(order)
        for k, idx in enumerate(order[: (len(order) // grad_accum) * grad_accum]):
            r = train_rows[idx]
            b = build_dpo_batch(tok, [r["prompt"]], [r["chosen"]], [r["rejected"]], max_length)
            with torch.autocast("cuda", dtype=torch.float16):
                pc, pr, rc, rr = policy_and_reference_logps(policy, reference, b)
                loss = dpo_loss_from_logps(pc, pr, rc, rr, beta)
            scaler.scale(loss / grad_accum).backward()
            m = dpo_metrics_from_logps(pc.detach(), pr.detach(), rc, rr, beta)
            acc_l += loss.item(); acc_a += m["reward_accuracy"]; acc_m += m["reward_margin"]; rn += 1
            if (k + 1) % grad_accum == 0:
                scaler.unscale_(opt)
                gn = torch.nn.utils.clip_grad_norm_(policy.parameters(), clip)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % log_every == 0:
                    d = {"step": step, "train_loss": acc_l / rn, "train_reward_acc": acc_a / rn,
                         "train_reward_margin": acc_m / rn, "grad_norm": float(gn),
                         "elapsed_s": time.time() - t0}
                    log(d); print(d)
                    acc_l = acc_a = acc_m = 0.0; rn = 0
                if step % eval_every == 0 or step == total:
                    run_eval(step)
    _save_fp16(policy, tok, os.path.join(out_dir, "final"))
    log({"done": True, "best_val_reward_acc": best_acc, "best_val_reward_margin": best_margin})

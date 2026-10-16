"""From-scratch DPO. Usage:
  python scripts/train_dpo.py --dataset hh --sft_path outputs/sft/best --out_dir outputs/dpo_hh
  python scripts/train_dpo.py --dataset ultrafeedback --sft_path outputs/sft/best --out_dir outputs/dpo_uf
"""
import argparse, json, os
from transformers import AutoTokenizer
from pa2 import data
from pa2.dpo import train_dpo, filter_long_prompts

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", choices=["hh", "ultrafeedback"], required=True)
ap.add_argument("--sft_path", required=True)
ap.add_argument("--out_dir", required=True)
ap.add_argument("--n_train", type=int, default=15000)
ap.add_argument("--n_val", type=int, default=1000)
ap.add_argument("--beta", type=float, default=0.1)
ap.add_argument("--lr", type=float, default=1e-6)
ap.add_argument("--grad_accum", type=int, default=32)
ap.add_argument("--max_length", type=int, default=512)
ap.add_argument("--max_prompt_tokens", type=int, default=256)
ap.add_argument("--eval_every", type=int, default=100)
ap.add_argument("--seed", type=int, default=42)
a = ap.parse_args()

loader = data.load_hh if a.dataset == "hh" else data.load_ultrafeedback
tok = AutoTokenizer.from_pretrained(a.sft_path)
# oversample, drop pairs with prompts > max_prompt_tokens, then split
pool, _ = loader(a.n_train + a.n_val + 6000, 0, a.seed)
pool = filter_long_prompts(pool, tok, a.max_prompt_tokens)
val, train = pool[:a.n_val], pool[a.n_val:a.n_val + a.n_train]
print(f"{len(train)} train / {len(val)} val pairs after prompt-length filter")
os.makedirs(a.out_dir, exist_ok=True)
json.dump(vars(a), open(os.path.join(a.out_dir, "run_config.json"), "w"), indent=2)
train_dpo(train, val, a.out_dir, a.sft_path, beta=a.beta, max_length=a.max_length,
          grad_accum=a.grad_accum, lr=a.lr, eval_every=a.eval_every, seed=a.seed)

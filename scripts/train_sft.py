import argparse, json, os
from pa2.data import load_sft
from pa2.sft import train_sft

ap = argparse.ArgumentParser()
ap.add_argument("--out_dir", required=True)
ap.add_argument("--eval_every", type=int, default=100)
ap.add_argument("--seed", type=int, default=42)
a = ap.parse_args()

train, val = load_sft(30000, 1000, a.seed)
print(len(train), "train /", len(val), "val")
os.makedirs(a.out_dir, exist_ok=True)
json.dump(vars(a), open(os.path.join(a.out_dir, "run_config.json"), "w"))
train_sft(train, val, a.out_dir, eval_every=a.eval_every, seed=a.seed)
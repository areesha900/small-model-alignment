"""Create the fixed AlpacaEval subset (300 prompts, seed 42)."""
import json, random, argparse, os
from datasets import load_dataset

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=300)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--out", default="eval_sets/alpaca_eval_300_seed42.json")
a = ap.parse_args()

try:
    from huggingface_hub import hf_hub_download
    ds = json.load(open(hf_hub_download("tatsu-lab/alpaca_eval", "alpaca_eval.json", repo_type="dataset")))
except Exception:
    ds = load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval", split="eval")
idx = list(range(len(ds)))
random.Random(a.seed).shuffle(idx)
rows = [{"id": f"alpaca_{i}", "instruction": ds[i]["instruction"]} for i in sorted(idx[:a.n])]
os.makedirs(os.path.dirname(a.out), exist_ok=True)
json.dump(rows, open(a.out, "w"), indent=2)
print(f"saved {len(rows)} prompts -> {a.out}")
"""Build randomized A/B pairwise JSONL for DPO vs SFT. Mapping A/B -> model is hidden metadata."""
import json, random, argparse

ap = argparse.ArgumentParser()
ap.add_argument("--sft_path", required=True)
ap.add_argument("--dpo_path", required=True)
ap.add_argument("--out_path", required=True)
ap.add_argument("--seed", type=int, default=42)
a = ap.parse_args()

sft = {r["id"]: r for r in json.load(open(a.sft_path))}
dpo = {r["id"]: r for r in json.load(open(a.dpo_path))}
rng = random.Random(a.seed)
with open(a.out_path, "w") as f:
    for k in sorted(sft.keys() & dpo.keys()):
        dpo_is_a = rng.random() < 0.5
        A, B = (dpo[k], sft[k]) if dpo_is_a else (sft[k], dpo[k])
        f.write(json.dumps({
            "id": k, "instruction": sft[k]["instruction"],
            "response_a": A["output"], "response_b": B["output"],
            "meta": {"a_is": "dpo" if dpo_is_a else "sft", "b_is": "sft" if dpo_is_a else "dpo"},
        }) + "\n")
print("wrote", a.out_path)

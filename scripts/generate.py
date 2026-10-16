"""Generate outputs from a checkpoint on a fixed eval file. Same decoding settings for every model!"""
import json, argparse, os, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pa2.data import format_prompt

ap = argparse.ArgumentParser()
ap.add_argument("--model_path", required=True)
ap.add_argument("--eval_path", required=True)
ap.add_argument("--out_path", required=True)
ap.add_argument("--generator", required=True)       # e.g. qwen2.5-0.5b-base / sft / dpo-ultrafeedback
ap.add_argument("--dataset", default="alpaca_eval")
ap.add_argument("--limit", type=int, default=None)  # use 3-5 for the sanity check
ap.add_argument("--batch_size", type=int, default=16)
ap.add_argument("--max_new_tokens", type=int, default=384)
ap.add_argument("--repetition_penalty", type=float, default=1.0)
ap.add_argument("--no_repeat_ngram_size", type=int, default=0)
a = ap.parse_args()

tok = AutoTokenizer.from_pretrained(a.model_path, padding_side="left")
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(a.model_path, torch_dtype=torch.float16).cuda().eval()

data = json.load(open(a.eval_path))
if a.limit:
    data = data[:a.limit]
done = {}
if os.path.exists(a.out_path):  # resume
    done = {r["id"]: r for r in json.load(open(a.out_path))}

todo = [d for d in data if d["id"] not in done]
for i in range(0, len(todo), a.batch_size):
    batch = todo[i:i + a.batch_size]
    prompts = [format_prompt(d["instruction"]) for d in batch]
    enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=256).to("cuda")
    with torch.no_grad():
        out = model.generate(**enc, do_sample=False, max_new_tokens=a.max_new_tokens,
                             repetition_penalty=a.repetition_penalty,
                             no_repeat_ngram_size=a.no_repeat_ngram_size,
                             pad_token_id=tok.pad_token_id)
    texts = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    for d, t in zip(batch, texts):
        done[d["id"]] = {"id": d["id"], "instruction": d["instruction"], "output": t.strip(),
                         "generator": a.generator, "dataset": a.dataset}
    json.dump(list(done.values()), open(a.out_path, "w"), indent=2)  # save after every batch
    print(f"{len(done)}/{len(data)} done")

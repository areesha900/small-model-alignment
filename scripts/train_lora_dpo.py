"""Task 5: TRL DPOTrainer + LoRA on the persona preference subset (SmolLM2-360M-Instruct).
Trains, saves the adapter, then generates base-vs-LoRA outputs on the style eval prompts.
Versions of TRL differ slightly in DPOConfig/DPOTrainer args - if one is rejected, check the
installed version's docs (this targets trl>=0.12,<1)."""
import argparse, json, os, torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/persona_lora/ramsay_style_preferences.json")
ap.add_argument("--eval_prompts", default="eval_sets/persona_style_eval_16.json")
ap.add_argument("--out_dir", default="outputs/lora_dpo")
ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-360M-Instruct")
ap.add_argument("--rank", type=int, default=16)
ap.add_argument("--alpha", type=int, default=32)
ap.add_argument("--dropout", type=float, default=0.05)
ap.add_argument("--lr", type=float, default=5e-5)
ap.add_argument("--beta", type=float, default=0.1)
ap.add_argument("--epochs", type=int, default=3)
ap.add_argument("--batch_size", type=int, default=4)
ap.add_argument("--grad_accum", type=int, default=4)
ap.add_argument("--seed", type=int, default=42)
a = ap.parse_args()

rows = json.load(open(a.data))
ds = Dataset.from_list([{
    "prompt": [{"role": "user", "content": r["prompt"]}],
    "chosen": [{"role": "assistant", "content": r["chosen"]}],
    "rejected": [{"role": "assistant", "content": r["rejected"]}]} for r in rows])

tok = AutoTokenizer.from_pretrained(a.model)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32)

lora = LoraConfig(r=a.rank, lora_alpha=a.alpha, lora_dropout=a.dropout, task_type="CAUSAL_LM",
                  target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
cfg = DPOConfig(output_dir=a.out_dir, beta=a.beta, learning_rate=a.lr, num_train_epochs=a.epochs,
                per_device_train_batch_size=a.batch_size, gradient_accumulation_steps=a.grad_accum,
                max_length=512, fp16=True, logging_steps=5, save_strategy="no", report_to="none",
                seed=a.seed)
trainer = DPOTrainer(model=model, args=cfg, train_dataset=ds, processing_class=tok, peft_config=lora)
trainer.train()
trainer.save_model(os.path.join(a.out_dir, "adapter"))
json.dump(trainer.state.log_history, open(os.path.join(a.out_dir, "log_history.json"), "w"), indent=2)
json.dump(vars(a), open(os.path.join(a.out_dir, "run_config.json"), "w"), indent=2)


@torch.no_grad()
def gen(m, prompt):
    text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                   add_generation_prompt=True)
    enc = tok(text, return_tensors="pt").to(m.device)
    out = m.generate(**enc, do_sample=False, max_new_tokens=200, pad_token_id=tok.pad_token_id)
    return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()


prompts = json.load(open(a.eval_prompts))
m = trainer.model.eval()
lora_out = [gen(m, p["instruction"]) for p in prompts]
with m.disable_adapter():  # same weights, adapter off = base model
    base_out = [gen(m, p["instruction"]) for p in prompts]
res = [{**p, "base_output": b, "lora_output": l} for p, b, l in zip(prompts, base_out, lora_out)]
json.dump(res, open(os.path.join(a.out_dir, "persona_generations.json"), "w"), indent=2)
print("saved", len(res), "generations")

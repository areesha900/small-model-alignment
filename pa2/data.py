"""Dataset loading + prompt formatting. One template used everywhere (SFT, DPO, eval)."""
import re
from datasets import load_dataset

TEMPLATE = "### Instruction:\n{instruction}\n\n### Response:\n"


def format_prompt(instruction: str) -> str:
    return TEMPLATE.format(instruction=instruction.strip())


def load_sft(n_train=30000, n_val=1000, seed=42):
    """UltraChat -> single-turn (prompt, response) pairs."""
    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
    ds = ds.shuffle(seed=seed)
    rows = []
    for ex in ds:
        msgs = ex["messages"]
        if len(msgs) < 2 or msgs[0]["role"] != "user" or msgs[1]["role"] != "assistant":
            continue
        rows.append({"prompt": format_prompt(msgs[0]["content"]), "response": msgs[1]["content"].strip()})
        if len(rows) >= n_train + n_val:
            break
    return rows[:n_train], rows[n_train:n_train + n_val]


def _split_hh(text):
    """HH-RLHF strings look like '\n\nHuman: ...\n\nAssistant: ...'. Return (prompt, reply) or None if multi-turn."""
    if text.count("\n\nHuman:") != 1:
        return None
    m = re.match(r"\n\nHuman: (.*?)\n\nAssistant: (.*)$", text, flags=re.S)
    return (m.group(1).strip(), m.group(2).strip()) if m else None


def load_hh(n_train=15000, n_val=1000, seed=42):
    ds = load_dataset("Anthropic/hh-rlhf", split="train").shuffle(seed=seed)
    rows = []
    for ex in ds:
        c, r = _split_hh(ex["chosen"]), _split_hh(ex["rejected"])
        if not c or not r or c[0] != r[0] or not c[1] or not r[1]:
            continue
        rows.append({"prompt": format_prompt(c[0]), "chosen": c[1], "rejected": r[1]})
        if len(rows) >= n_train + n_val:
            break
    return rows[:n_train], rows[n_train:n_train + n_val]


def load_ultrafeedback(n_train=15000, n_val=1000, seed=42):
    ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs").shuffle(seed=seed)
    rows = []
    for ex in ds:
        ch, rj = ex["chosen"], ex["rejected"]  # lists of {"role","content"}
        if len(ch) != 2 or len(rj) != 2:
            continue
        rows.append({"prompt": format_prompt(ex["prompt"]),
                     "chosen": ch[1]["content"].strip(), "rejected": rj[1]["content"].strip()})
        if len(rows) >= n_train + n_val:
            break
    return rows[:n_train], rows[n_train:n_train + n_val]

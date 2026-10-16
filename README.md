# Small-Model Alignment Starter

This branch is the student starter package for the SFT + DPO alignment assignment. It is intentionally minimal: it provides the tested environment metadata, assignment PDF, fixed evaluation/data files, and no completed training implementation.

## What Is Included

```text
requirements.txt
pyproject.toml
submission_adapter.py
tests/test_public_adapter_smoke.py
tests/toy_lm.py
docs/assignment_manual.pdf
data/persona_lora/ramsay_style_preferences.json
data/persona_lora/README.md
eval_sets/hh_rlhf_eval_300_seed42.json
```

## Recommended Kaggle Setup

Use a Kaggle notebook with the `GPU T4 x2` accelerator. After cloning this repository/branch, run:

```bash
python -m pip uninstall -y -q torchao || true
python -m pip install -q "transformers>=4.43.0,<5" "datasets>=2.18,<5" huggingface_hub accelerate tqdm pyyaml sentencepiece safetensors pytest
python -m pip install -q -e . --no-deps
```

This keeps Kaggle's preinstalled CUDA-compatible PyTorch intact, avoids known stale `torchao` / PEFT conflicts, and makes the local repository importable without replacing the GPU stack.

## What You Need To Build

You will add your own training and evaluation code on top of this starter. At minimum, your project should implement:

- SFT training on 30,000 examples.
- From-scratch DPO on HH-RLHF-style preference data.
- From-scratch DPO on UltraFeedback-style preference data.
- TRL `DPOTrainer` + LoRA training on the provided persona preference data.
- Generation and DPO-vs-SFT evaluation outputs.
- A short final report with plots and analysis.

Read `docs/assignment_manual.pdf` for the full requirements and suggested workflow.

## Submission Adapter And Public Tests

Keep the six function signatures in `submission_adapter.py` unchanged. Implement
each function as a small wrapper around your own SFT/DPO code. The grader uses
this stable interface so your internal file and class organization can remain
your choice.

Run the public tests from the repository root:

```bash
python -m pip install -q pytest
PYTHONPATH=. python -m pytest -q tests/test_public_adapter_smoke.py
```

The tests use a tiny local tokenizer and model, so they do not download a model,
load the assignment datasets, or require a GPU. The starter adapter raises
`NotImplementedError` until you connect it to your implementation. Passing the
public tests checks the interface and a few basic invariants; the private grader
also checks numerical correctness and harder edge cases described in the manual.

For `compute_sft_loss`, the assignment accepts either the mean of per-example
summed response negative log probabilities or the mean over all unmasked response
tokens. Use one reduction consistently and document it in the report.

## Fixed Files

- `eval_sets/hh_rlhf_eval_300_seed42.json` is the fixed synthetic helpful/harmless evaluation set.
- For AlpacaEval-style evaluation, use 300 prompts with seed `42` so comparisons are consistent.
- `data/persona_lora/ramsay_style_preferences.json` is the small preference dataset for the LoRA-DPO persona task.

## Saving Work On Kaggle

Kaggle sessions are temporary. Write important outputs under `/kaggle/working`, save notebook versions, and archive checkpoints/generations as zip files or Kaggle output datasets before ending a session.

---

# Implementation and reproduction (CS 5326, PA2)

This repository adds my SFT, DPO, LoRA-DPO, generation and judging code on top of the starter package above.
Checkpoints are **not** stored here (size); they are kept in the committed Kaggle notebook outputs of the SFT, DPO and LoRA notebooks listed in the run order below.

## Code

| What | Where |
|---|---|
| SFT (response-only masking, token-mean loss, own training loop) | `pa2/sft.py`, `scripts/train_sft.py` |
| From-scratch DPO (loss, metrics, frozen-reference loop, best checkpoint by validation reward accuracy) | `pa2/dpo.py`, `scripts/train_dpo.py` |
| Data loading and the single prompt template | `pa2/data.py` |
| TRL `DPOTrainer` + LoRA persona task | `scripts/train_lora_dpo.py` |
| Generation, A/B pair building, judging | `scripts/generate.py`, `scripts/gen_alpaca_subset.py`, `scripts/make_pairs.py`, `scripts/judge_pairwise_kaggle.py` |
| Grader interface | `submission_adapter.py` (public tests: `PYTHONPATH=. python -m pytest -q tests/test_public_adapter_smoke.py`) |

## Run order (Kaggle, GPU T4 x2 for training and generation, no GPU for judging)

| Step | Notebook (`notebooks/`) | Produces |
|---|---|---|
| 1. setup, smoke tests, AlpacaEval subset, base generation | `genai-pa2.ipynb` | `eval_sets/alpaca_eval_300_seed42.json`, `outputs/gen/base_alpaca.json` |
| 2. SFT (30k examples, 937 steps) | `pa2-sft.ipynb` | `outputs/sft/{best,final}`, `metrics.jsonl` |
| 3. DPO on HH-RLHF and UltraFeedback (468 steps each) | `pa2-dpo.ipynb` | `outputs/dpo_hh/`, `outputs/dpo_uf/` (`best`, `final`, `metrics.jsonl`) |
| 4. LoRA-DPO persona variants | `pa2-lora.ipynb` | `outputs/lora_dpo_<variant>/` (adapter, `log_history.json`, `persona_generations*.json`) |
| 5. generation + A/B pairs for DPO vs SFT | `pa2-gen.ipynb` | `outputs/gen/*.json`, `outputs/eval/{uf,hh}_pairs.jsonl` |
| 6. judging, pass 1 (random order) | `pa2-judge.ipynb` | `outputs/eval/{uf,hh}_judged.jsonl`, `*_summary.json` |
| 7. judging, swapped-order pass and LoRA persona | `pa2-judge2.ipynb` | `outputs/eval/*_swapped*`, `two_order_summary.json`, `persona_*` |

All seeds are 42. Generation uses greedy decoding with `--max_new_tokens 384 --repetition_penalty 1.1 --no_repeat_ngram_size 0` for every model.
Judge: `google/gemini-3.6-flash` through `kaggle_benchmarks` (5 pairs per request, 5 s between requests, 3 attempts per batch).

## Where the reported numbers come from

- Training curves: `outputs/sft/metrics.jsonl`, `outputs/dpo_hh/metrics.jsonl`, `outputs/dpo_uf/metrics.jsonl`
- DPO vs SFT tables: `outputs/eval/uf_judged_summary.json`, `hh_judged_summary.json`, `*_swapped_summary.json`, `two_order_summary.json`
- LoRA table: `outputs/eval/persona_summary.json`, `outputs/lora_dpo_*/log_history.json`
- Report and figures: `report/PA2_report.pdf`, `report/figures/`


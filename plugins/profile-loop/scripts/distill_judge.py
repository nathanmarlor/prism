#!/usr/bin/env python3
"""Distil a small, specialised CPU judge for your rubric.

The off-the-shelf CPU judge (Qwen2.5-1.5B via llama.cpp) works out of the box.
This script produces something better for one specific task: a smaller model
fine-tuned on your own preference labels, so it judges your rubric more
accurately and runs even faster on every request. This is the "distil it into a
dedicated SLM judge" idea from the concept, made concrete.

The pipeline has four stages. Only stage 2 needs a training machine; the rest
run anywhere.

  1. Build a training set of judge examples from preference pairs.
  2. Fine-tune a tiny base model (LoRA) to output 'A'/'B' for the better response.
  3. Merge, convert to GGUF, and quantise to Q4_K_M with llama.cpp.
  4. Point the plugin at the result:  PROFILE_LOOP_MODEL_PATH=judge.gguf

Input: a JSONL file of preference pairs, one per line:
    {"prompt": "...", "chosen": "...", "rejected": "..."}
These can come from your validation labels, from a stronger teacher judge
labelling sampled responses, or from real corrections logged by the loop.

This file is a runnable recipe, not a dependency of the plugin. Install the
training extras first:  pip install "transformers>=4.44" peft trl datasets
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


JUDGE_PROMPT = (
    "You are a strict evaluator. Rubric: {rubric}\n\n"
    "User prompt:\n{prompt}\n\n"
    "Response A:\n{a}\n\nResponse B:\n{b}\n\n"
    "Which response better fits the rubric? Answer with only 'A' or 'B'."
)


def build_dataset(pairs_path: str, rubric: str, out_path: str) -> int:
    """Stage 1: turn preference pairs into balanced judge examples.

    Each pair becomes two examples with the chosen response in each position,
    so the model learns the rubric rather than a position habit.
    """
    rows = []
    with open(pairs_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            chosen, rejected = ex["chosen"], ex["rejected"]
            # chosen as A
            rows.append({"prompt": JUDGE_PROMPT.format(
                rubric=rubric, prompt=ex["prompt"], a=chosen, b=rejected),
                "completion": "A"})
            # chosen as B (same pair, swapped) -> teaches position invariance
            rows.append({"prompt": JUDGE_PROMPT.format(
                rubric=rubric, prompt=ex["prompt"], a=rejected, b=chosen),
                "completion": "B"})
    random.shuffle(rows)
    with open(out_path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows)


def train(dataset_path: str, base_model: str, out_dir: str) -> None:
    """Stage 2: LoRA fine-tune the base model to emit A/B. Needs a GPU ideally,
    but a small base (<=1.5B) trains on CPU in a pinch for a few hundred examples.
    """
    from datasets import load_dataset
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    ds = load_dataset("json", data_files=dataset_path, split="train")

    def to_text(ex):
        return {"text": ex["prompt"] + ex["completion"]}

    ds = ds.map(to_text)
    peft_cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                          target_modules="all-linear", task_type="CAUSAL_LM")
    cfg = SFTConfig(output_dir=out_dir, num_train_epochs=3,
                    per_device_train_batch_size=8, learning_rate=2e-4,
                    max_length=1024, logging_steps=10, save_strategy="epoch")
    trainer = SFTTrainer(model=base_model, train_dataset=ds,
                         peft_config=peft_cfg, args=cfg,
                         dataset_text_field="text")
    trainer.train()
    trainer.save_model(out_dir)
    print(f"Saved adapter to {out_dir}")


def gguf_instructions(out_dir: str) -> str:
    """Stage 3: the llama.cpp steps to get a CPU-ready GGUF."""
    return f"""
Stage 3 — merge, convert, quantise (run in a llama.cpp checkout):

  # merge the LoRA into the base weights
  python -c "from peft import AutoPeftModelForCausalLM as M; \\
             m=M.from_pretrained('{out_dir}').merge_and_unload(); \\
             m.save_pretrained('{out_dir}/merged')"

  # convert to GGUF, then quantise to 4-bit
  python convert_hf_to_gguf.py {out_dir}/merged --outfile judge-f16.gguf --outtype f16
  ./llama-quantize judge-f16.gguf judge-q4_k_m.gguf Q4_K_M

Stage 4 — use it:

  export PROFILE_LOOP_JUDGE=local
  export PROFILE_LOOP_MODEL_PATH=$(pwd)/judge-q4_k_m.gguf
  # then re-run /profile-loop:validate to confirm the distilled judge still
  # agrees with your labels before trusting it.
"""


def main() -> None:
    ap = argparse.ArgumentParser(description="Distil a CPU judge for your rubric.")
    ap.add_argument("pairs", help="JSONL of {prompt, chosen, rejected}")
    ap.add_argument("--rubric", required=True, help="the rubric string from your evaluator spec")
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-0.5B-Instruct",
                    help="tiny base to distil into (smaller = faster on CPU)")
    ap.add_argument("--dataset", default="judge_dataset.jsonl")
    ap.add_argument("--out-dir", default="judge-lora")
    ap.add_argument("--train", action="store_true", help="run stage 2 (needs training deps)")
    args = ap.parse_args()

    n = build_dataset(args.pairs, args.rubric, args.dataset)
    print(f"Stage 1: wrote {n} judge examples to {args.dataset}")

    if args.train:
        train(args.dataset, args.base_model, args.out_dir)
        print(gguf_instructions(args.out_dir))
    else:
        print("Stage 2 skipped (pass --train once training deps are installed).")
        print(gguf_instructions(args.out_dir))


if __name__ == "__main__":
    main()

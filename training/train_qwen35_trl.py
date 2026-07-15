#!/usr/bin/env python3
"""
TRL SFT + PEFT LoRA fine-tune for Qwen3.5-35B-A3B (MoE, BF16) on DGX Spark GB10.

Architecture notes:
  - 40 layers total: 30 linear-attention + 10 full-attention (every 4th layer)
  - 256 MoE experts, 8 active per token, hidden_size=2048, 35B params
  - SDPA attention (GB10 doesn't support FlashAttention — SM121 kernel gap)
  - LoRA targets q/k/v/o_proj in the 10 full-attention layers only;
    PEFT silently skips the linear-attention layers which lack those modules.

Single-node run (inside nemo:25.09 container):
  docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \\
    --network host \\
    -v /home/cognitivo/local-llm-setup/models:/models \\
    -v /home/cognitivo/finagent-finetune:/workspace \\
    nvcr.io/nvidia/nemo:25.09 \\
    bash -c "pip install -q 'trl>=0.13' 'transformers>=5.0.0' datasets && python /workspace/training/train_qwen35_trl.py"

Requires transformers>=5.0 for Qwen3_5MoeForConditionalGeneration support.
"""

import os, json, torch
from datasets import Dataset
from transformers import AutoTokenizer, BitsAndBytesConfig, Qwen3_5MoeForConditionalGeneration
from peft import LoraConfig, TaskType
from trl import SFTTrainer, SFTConfig

# ─── Config ──────────────────────────────────────────────────────────────────

MODEL_PATH  = os.environ.get("MODEL_PATH",  "/models/Qwen3.5-35B-A3B")
TRAIN_FILE  = os.environ.get("TRAIN_FILE",  "/workspace/data/train.jsonl")
VAL_FILE    = os.environ.get("VAL_FILE",    "/workspace/data/val.jsonl")
OUTPUT_DIR  = os.environ.get("OUTPUT_DIR",  "/models/checkpoints/qwen35-finance-lora")
MAX_SEQ_LEN = int(os.environ.get("MAX_SEQ_LEN",  "512"))
BATCH_SIZE  = int(os.environ.get("BATCH_SIZE",   "1"))
GRAD_ACCUM  = int(os.environ.get("GRAD_ACCUM",   "4"))
MAX_STEPS   = int(os.environ.get("MAX_STEPS",    "500"))
LR          = float(os.environ.get("LR",         "1e-4"))
LORA_RANK   = int(os.environ.get("LORA_RANK",    "16"))
USE_4BIT    = os.environ.get("USE_4BIT", "0") != "0"

# Qwen3.5 uses ChatML format
TEMPLATE = (
    "<|im_start|>user\n{input}<|im_end|>\n"
    "<|im_start|>assistant\n{output}<|im_end|>"
)


# ─── Dataset ─────────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> Dataset:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            s = json.loads(line)
            records.append({
                "text": TEMPLATE.format(input=s["input"], output=s["output"])
            })
    return Dataset.from_list(records)


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Loading tokenizer from {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    bnb_config = None
    if USE_4BIT:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    print(f"Loading model (BF16={'4bit' if USE_4BIT else 'full'}, attn=sdpa)...")
    # Must use Qwen3_5MoeForConditionalGeneration (not AutoModelForCausalLM).
    # The saved weights are under model.language_model.* + mtp.* prefix (VLM+MTP layout).
    # AutoModelForCausalLM resolves to Qwen3_5MoeForCausalLM which expects model.layers.*
    # and causes a deterministic crash at shard 2 during weight loading.
    model = Qwen3_5MoeForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=True,
        device_map="auto",
        low_cpu_mem_usage=True,
        quantization_config=bnb_config,
    )
    model.config.use_cache = False

    # LoRA applies only to full-attention layers (layers 3,7,11,...,39).
    # PEFT skips linear-attention layers that lack q/k/v/o_proj — no special handling needed.
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_RANK,
        lora_alpha=LORA_RANK * 2,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        modules_to_save=None,
    )

    print("Loading datasets...")
    train_ds = load_jsonl(TRAIN_FILE)
    val_ds   = load_jsonl(VAL_FILE)
    print(f"  train: {len(train_ds)}  val: {len(val_ds)}")

    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        max_steps=MAX_STEPS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        lr_scheduler_type="cosine",
        warmup_steps=50,
        bf16=True,
        fp16=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch_fused",
        max_grad_norm=1.0,
        logging_steps=10,
        save_steps=max(MAX_STEPS // 5, 50),
        save_total_limit=2,
        eval_strategy="steps",
        eval_steps=max(MAX_STEPS // 5, 50),
        load_best_model_at_end=False,
        max_seq_length=MAX_SEQ_LEN,
        dataset_num_proc=4,
        dataloader_num_workers=4,
        report_to="none",
        run_name="qwen35-finance-lora",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        peft_config=lora_config,
        processing_class=tokenizer,
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {n_params/1e6:.1f}M  |  steps: {MAX_STEPS}  |  rank: {LORA_RANK}")
    print(f"Starting training...")

    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\nDone. Adapter saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

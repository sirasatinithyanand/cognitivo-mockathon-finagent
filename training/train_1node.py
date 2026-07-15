#!/usr/bin/env python3
"""
NeMo 2.x LoRA SFT for Llama-3.1-Nemotron-Nano-8B-v1.
Uses HFAutoModelForCausalLM + llm.api.finetune — no checkpoint conversion required.
Run via scripts/03_train_1node.sh (sets env vars and launches the NeMo container).
"""

import os, json, torch
import fiddle as fdl
import lightning.pytorch as pl
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from nemo import lightning as nl
from nemo.collections import llm
from nemo.collections.llm.recipes.optim.adam import pytorch_adam_with_cosine_annealing

# ─── Config ──────────────────────────────────────────────────────────────────

MODEL_PATH  = os.environ.get("MODEL_PATH",  "/models/Llama-3.1-Nemotron-Nano-8B-v1")
TRAIN_FILE  = os.environ.get("TRAIN_FILE",  "/workspace/data/train.jsonl")
VAL_FILE    = os.environ.get("VAL_FILE",    "/workspace/data/val.jsonl")
OUTPUT_DIR  = os.environ.get("OUTPUT_DIR",  "/models/checkpoints/nemotron-finance-lora")
MAX_SEQ_LEN = int(os.environ.get("MAX_SEQ_LEN",  "512"))   # 512: 16x less eager-attention cost vs 2048
BATCH_SIZE  = int(os.environ.get("BATCH_SIZE",   "1"))
GRAD_ACCUM  = int(os.environ.get("GRAD_ACCUM",   "4"))
MAX_STEPS   = int(os.environ.get("MAX_STEPS",    "200"))
LR          = float(os.environ.get("LR",         "5e-5"))
LORA_RANK   = int(os.environ.get("LORA_RANK",    "32"))


# ─── Dataset ─────────────────────────────────────────────────────────────────

TEMPLATE = (
    "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
    "{input}"
    "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    "{output}"
    "<|eot_id|>"
)


def tokenize_file(path, tokenizer, max_len):
    """Pre-tokenize a JSONL file. Called before llm.api.finetune can swap the tokenizer."""
    records = []
    pad_id = tokenizer.pad_token_id
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            s = json.loads(line)
            text = TEMPLATE.format(input=s["input"], output=s["output"])
            enc = tokenizer(
                text,
                max_length=max_len,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].squeeze(0)

            prompt = TEMPLATE.split("{output}")[0].format(input=s["input"])
            prompt_ids = tokenizer(
                prompt, add_special_tokens=False, return_tensors="pt"
            )["input_ids"].squeeze(0)
            prompt_len = min(len(prompt_ids), max_len)

            # NeMo's fused_linear_cross_entropy uses shift=False, so labels must be
            # pre-shifted left by 1: labels[t] = input_ids[t+1] (predict next token).
            labels = torch.full((max_len,), -100, dtype=input_ids.dtype)
            labels[:-1] = input_ids[1:]

            # loss_mask also shifts left by 1 to stay aligned with labels.
            loss_mask = torch.zeros(max_len, dtype=torch.float)
            loss_mask[prompt_len:-1] = 1.0
            loss_mask[labels == -100] = 0.0
            loss_mask[labels == pad_id] = 0.0

            records.append({
                "input_ids":      input_ids,
                "labels":         labels,
                "loss_mask":      loss_mask,
                "attention_mask": enc["attention_mask"].squeeze(0),
            })
    return records


class FinanceDataset(Dataset):
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return self.records[idx]


class FinanceDataModule(pl.LightningDataModule):
    def __init__(self, tokenizer):
        super().__init__()
        self.micro_batch_size  = BATCH_SIZE
        self.global_batch_size = BATCH_SIZE
        self.seq_length        = MAX_SEQ_LEN   # NeMo 25.09 OneLogger callback needs this
        # Tokenize now, before llm.api.finetune replaces data.tokenizer
        print("Pre-tokenizing train split...")
        self._train = FinanceDataset(tokenize_file(TRAIN_FILE, tokenizer, MAX_SEQ_LEN))
        print(f"  {len(self._train)} samples ready")
        print("Pre-tokenizing val split...")
        self._val = FinanceDataset(tokenize_file(VAL_FILE, tokenizer, MAX_SEQ_LEN))
        print(f"  {len(self._val)} samples ready")

    def train_dataloader(self):
        return DataLoader(self._train, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    def val_dataloader(self):
        return DataLoader(self._val, batch_size=BATCH_SIZE, num_workers=0)


# ─── Model ───────────────────────────────────────────────────────────────────
# Must be module-level so Fiddle can serialize it by import path at checkpoint time.

LOAD_4BIT = os.environ.get("LOAD_4BIT", "0") != "0"  # disabled: bitsandbytes not in nemo:25.04


class QLoRAAutoModel(llm.HFAutoModelForCausalLM):
    """Enables input require grads so gradient checkpointing works with frozen base weights."""
    def configure_model(self):
        super().configure_model()
        if self.model is not None:
            self.model.enable_input_require_grads()


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Loading tokenizer from {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, trust_remote_code=True, use_fast=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Building NeMo 2.x HF model wrapper (4-bit={LOAD_4BIT}, attn=sdpa)...")
    model = QLoRAAutoModel(
        model_name=MODEL_PATH,
        trust_remote_code=True,
        default_dtype=torch.bfloat16,
        enable_grad_ckpt=False,
        load_in_4bit=LOAD_4BIT,
        attn_implementation="sdpa",
    )

    data = FinanceDataModule(tokenizer)

    # SingleDeviceStrategy saves only LoRA adapter weights
    strategy = pl.strategies.SingleDeviceStrategy(
        device="cuda:0",
        checkpoint_io=model.make_checkpoint_io(adapter_only=True),
    )

    nemo_logger = nl.NeMoLogger(
        name="finance-lora",
        log_dir=OUTPUT_DIR,
        use_datetime_version=False,
        ckpt=nl.ModelCheckpoint(
            save_last=True,
            every_n_train_steps=max(MAX_STEPS // 5, 20),
            monitor="reduced_train_loss",
            save_top_k=5,
        ),
    )

    optimizer = fdl.build(
        pytorch_adam_with_cosine_annealing(max_lr=LR, warmup_steps=50)
    )

    print(f"Starting training — {MAX_STEPS} steps, LR={LR}, LoRA rank={LORA_RANK}")
    llm.api.finetune(
        model=model,
        data=data,
        trainer=nl.Trainer(
            devices=1,
            max_steps=MAX_STEPS,
            accelerator="gpu",
            strategy=strategy,
            log_every_n_steps=10,
            val_check_interval=200,
            limit_val_batches=5,
            num_sanity_val_steps=0,
            accumulate_grad_batches=GRAD_ACCUM,
            gradient_clip_val=1.0,
            precision="bf16-mixed",
        ),
        optim=optimizer,
        log=nemo_logger,
        peft=llm.peft.LoRA(
            target_modules=["*_proj"],
            dim=LORA_RANK,
            dropout=0.05,
        ),
        resume=nl.AutoResume(
            resume_if_exists=True,
            resume_ignore_no_checkpoint=True,
        ),
    )
    print(f"Training complete. Checkpoints saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

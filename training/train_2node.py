#!/usr/bin/env python3
"""
NeMo 2.x LoRA SFT — FSDP2 strategy for 2-node DGX Spark cluster.
Each node holds ~50 GB of model shards; ~78 GB free per node for activations.

Run via torchrun (see 03_train_2node.sh):
  torchrun --nnodes=2 --nproc_per_node=1 ... 03_train_nemo2_fsdp2.py
"""

import os, json, torch
import fiddle as fdl
import lightning.pytorch as pl
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from nemo import lightning as nl
from nemo.collections import llm
from nemo.collections.llm.recipes.optim.adam import pytorch_adam_with_cosine_annealing


class BF16LoRAModel(llm.HFAutoModelForCausalLM):
    """Enables input require-grads so gradient checkpointing works with frozen base weights."""
    def configure_model(self):
        super().configure_model()
        if self.model is not None:
            self.model.enable_input_require_grads()

# ─── Config ──────────────────────────────────────────────────────────────────

MODEL_PATH  = os.environ.get("MODEL_PATH",  "/models/Llama-3.1-Nemotron-70B-Instruct-HF")
TRAIN_FILE  = os.environ.get("TRAIN_FILE",  "/workspace/data/train.jsonl")
VAL_FILE    = os.environ.get("VAL_FILE",    "/workspace/data/val.jsonl")
OUTPUT_DIR  = os.environ.get("OUTPUT_DIR",  "/models/checkpoints/nemotron-finance-lora")
MAX_SEQ_LEN = int(os.environ.get("MAX_SEQ_LEN",  "512"))   # 512 proven on GB10; 2048 OOMs with eager attn + 49B
BATCH_SIZE  = int(os.environ.get("BATCH_SIZE",   "1"))
GRAD_ACCUM  = int(os.environ.get("GRAD_ACCUM",   "8"))
MAX_STEPS   = int(os.environ.get("MAX_STEPS",    "500"))
LR          = float(os.environ.get("LR",         "1e-4"))
LORA_RANK   = int(os.environ.get("LORA_RANK",    "32"))
NUM_NODES   = int(os.environ.get("WORLD_SIZE",   "2"))


# ─── Dataset ─────────────────────────────────────────────────────────────────

TEMPLATE = (
    "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
    "{input}"
    "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    "{output}"
    "<|eot_id|>"
)


def tokenize_file(path, tokenizer, max_len):
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
                text, max_length=max_len, truncation=True,
                padding="max_length", return_tensors="pt",
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
            # Suppress loss on: prompt positions, padding, and the final position
            # (which has no next token to predict).
            loss_mask = torch.zeros(max_len, dtype=torch.float)
            loss_mask[prompt_len:-1] = 1.0
            loss_mask[labels == -100] = 0.0
            loss_mask[labels == pad_id] = 0.0
            records.append({
                "input_ids": input_ids, "labels": labels,
                "loss_mask": loss_mask,
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
        self.global_batch_size = BATCH_SIZE * NUM_NODES
        self.seq_length        = MAX_SEQ_LEN   # NeMo 25.09 OneLogger callback needs this
        print("Pre-tokenizing train split...")
        self._train = FinanceDataset(tokenize_file(TRAIN_FILE, tokenizer, MAX_SEQ_LEN))
        print("Pre-tokenizing val split...")
        self._val = FinanceDataset(tokenize_file(VAL_FILE, tokenizer, MAX_SEQ_LEN))

    def train_dataloader(self):
        return DataLoader(self._train, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    def val_dataloader(self):
        return DataLoader(self._val, batch_size=BATCH_SIZE, num_workers=0)


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    print(f"[rank {local_rank}] Loading tokenizer from {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, trust_remote_code=True, use_fast=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[rank {local_rank}] Building NeMo 2.x HF model wrapper (FSDP2, attn=sdpa)...")
    model = BF16LoRAModel(
        model_name=MODEL_PATH,
        trust_remote_code=True,
        default_dtype=torch.bfloat16,
        enable_grad_ckpt=True,
        attn_implementation="sdpa",
    )

    data = FinanceDataModule(tokenizer)

    # FSDP2: shards model across all ranks — each GB10 node holds ~50 GB
    strategy = nl.FSDP2Strategy(
        data_parallel_size=NUM_NODES,
        tensor_parallel_size=1,
        checkpoint_io=model.make_checkpoint_io(adapter_only=True),
    )

    nemo_logger = nl.NeMoLogger(
        name="finance-lora",
        log_dir=OUTPUT_DIR,
        use_datetime_version=False,
        ckpt=nl.ModelCheckpoint(
            save_last=True,
            every_n_train_steps=max(MAX_STEPS // 10, 50),
            monitor="reduced_train_loss",
            save_top_k=3,
        ),
    )

    optimizer = fdl.build(
        pytorch_adam_with_cosine_annealing(max_lr=LR, warmup_steps=50)
    )

    print(f"[rank {local_rank}] Starting FSDP2 training — {MAX_STEPS} steps, LR={LR}, LoRA rank={LORA_RANK}")
    llm.api.finetune(
        model=model,
        data=data,
        trainer=nl.Trainer(
            devices=1,
            num_nodes=NUM_NODES,
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
    print(f"[rank {local_rank}] Training complete. Checkpoints saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

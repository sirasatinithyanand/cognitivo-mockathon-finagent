# 03 — Fine-tuning Team (You): LoRA SFT of Nemotron-8B

You fine-tune the supplied **Llama-3.1-Nemotron-Nano-8B-v1** into the `domain-ft` model,
serve it on node1, and register it into the agentic team's LiteLLM proxy. Read
`01_Environment_and_Gotchas.md` and `04_Seam_Contract.md` first — the seam defines the exact
prompt format you must train against.

**Scope reality check:** `domain-ft` is used for **only the 3 AFR sentiment questions**
(MHQ058/067/080). Your job is a model that, given the AFR prompt format, reliably outputs a
**sentiment word (positive/negative/mixed) + market direction** — NOT numbers. The exact
counts/returns are computed by the agentic team's Python tools. Train tight to that task;
don't over-invest.

## Your files/scripts

```
FT=~/Cognitivo_Training/finagent-finetune-participant
$FT/scripts/01_prepare_data.py       # build JSONL training pairs from AFR/ASX/RBA
$FT/scripts/02_smoke_test.sh         # Llama-3.2-1B, 50 steps — validate the pipeline (~5 min)
$FT/scripts/03_train_1node.sh        # LoRA SFT Nemotron-8B (recommended baseline)
$FT/scripts/08_serve_domain_ft.sh    # serve trained adapter via vLLM on :8001
$FT/scripts/05_evaluate.py           # base vs finetuned on test set (evidence)
$FT/training/{train_1node.py,lora_finance.yaml}
$FT/data/{train,val,test}.jsonl      # 48k/6k/6k already built — but a GENERAL set (re-weight, see Step 1)
```

## Reference baseline (confirmed working on GB10)

| Param | Value |
|-------|-------|
| Container | `nvcr.io/nvidia/nemo:25.09` (25.04 crashes on first NCCL kernel) |
| LoRA rank | 32 |
| Seq length | 512 (longer may OOM on single node; use 256 if OOMing) |
| Learning rate | 5e-5 (1e-4 causes a loss spike at warmup step 50) |
| Steps | 100 full run; step 20 checkpoint already shows meaningful improvement |

## Step 0 — Cluster up (see `cluster-serving-stack` notes)
Bring up node1 for training/serving. Use `sg docker -c 'docker ...'` (docker-group quirk).
Node0's Qwen brain + LiteLLM are the agentic team's job — coordinate so you can register
`domain-ft` into their proxy in Step 4.

## Step 1 — Prepare training data (focus on the sentiment format)
The pre-built `$FT/data/*.jsonl` is a general instruction set. Regenerate (or top up) so the
**AFR-sentiment format from `04_Seam_Contract.md` dominates** — headline + article + RBA rate
→ sentiment + direction. Fix the dataset path (Desktop, not Downloads):

```bash
cd $FT
python scripts/01_prepare_data.py \
  --afr_dir  "/home/cognitivo/Desktop/HackathonDataset/Jsonl format DataSets/AFR Jasonl" \
  --asx_dir  "/home/cognitivo/Desktop/HackathonDataset/Jsonl format DataSets/ASX-18-companies-2015-2021-Jasonl" \
  --rba_file "/home/cognitivo/Desktop/HackathonDataset/Jsonl format DataSets/RBA-Rates-2010-2026/RBA-rates.jsonl" \
  --out_dir  data/
```
Sanity-check that a good fraction of `train.jsonl` prompts match the AFR sentiment template
(`Date: … / RBA cash rate: …% / AFR Headline: … / Article: … / assess the market sentiment`).

## Step 2 — Smoke test the pipeline (~5 min)
```bash
bash scripts/02_smoke_test.sh        # Llama-3.2-1B, 50 steps — confirms container/GPU/data/checkpoint saving
```

## Step 3 — Train (in tmux — earlyoom kills python otherwise)
```bash
tmux new-session -s finetune "bash scripts/03_train_1node.sh"
tail -f /tmp/nemo_1node.log
```
Key envs (defaults are the baseline above):
`MODEL_PATH=/models/Llama-3.1-Nemotron-Nano-8B-v1`, `OUTPUT_DIR=/models/checkpoints/nemotron8b-finance-lora`,
`MAX_STEPS=100`, `MAX_SEQ_LEN=512`, `LORA_RANK=32`, `LR=5e-5`.
Watch the loss curve; a checkpoint lands at step 20 (nothing before that — don't kill early).

## Step 4 — Export adapter + serve on node1 + register alias
```bash
ADAPTER_CHECKPOINT=/models/checkpoints/nemotron8b-finance-lora/finance-lora/checkpoints/last/hf_adapter \
bash scripts/08_serve_domain_ft.sh          # vLLM + LoRA on node1 :8001
```
Then have the agentic team add (or confirm) the `domain-ft` entry in `~/litellm/config.yaml`
pointing at `http://10.0.1.11:8001/v1`, and verify:
```bash
curl http://10.0.1.10:4000/v1/models        # domain-ft must be listed
```

## Step 5 — Validate the model in isolation (no agent needed)
Hit `domain-ft` directly with the exact AFR prompts for the 3 questions (see the table in
`04_Seam_Contract.md`) and confirm the output contains the right sentiment + direction:
```bash
curl -s http://10.0.1.10:4000/v1/chat/completions -H 'content-type: application/json' -d '{
  "model":"domain-ft",
  "messages":[{"role":"user","content":"Date: 2021-02-23\nRBA cash rate: 0.10%\nAFR Headline: Travel stocks take off on vaccine rollout\nArticle: ...\n\nAs an Australian financial analyst, assess the market sentiment and likely ASX impact."}],
  "max_tokens":250, "temperature":0.1 }' | python -m json.tool
# expect: positive sentiment + upward direction for ASX travel shares
```
Target reads: MHQ058 → positive / travel up · MHQ067 → mixed-negative / broad ASX
mixed-to-down · MHQ080 → positive / energy up.

## Step 6 — Evidence + submission (your deliverables)
```bash
python scripts/05_evaluate.py --test_file data/test.jsonl --base_model nemotron-base --ft_model domain-ft
```
Put into `training/`: training script(s), the data-prep command + notes, `lora_finance.yaml`
/ hyperparams, the training log, base-vs-FT metrics, and a short **model card** (what/why/how).
Fill the `model` block of `submission.json` (include only if reachable for review):
```json
"model": { "endpoint": "http://10.0.1.11:8001/v1", "model_name": "domain-ft" }
```

## Gotchas (yours specifically)
- `nemo:25.09+` only. Always `tmux`. `LR=5e-5` not `1e-4`. `MAX_SEQ_LEN=512` (256 if OOM).
- No checkpoint before step 20 — a crash before then restarts from scratch.
- The model outputs **text sentiment, not numbers** — don't train it to emit returns/prices.
- **Do not change the `sentiment_assess` prompt format** without telling the agentic team;
  if they prompt differently than you trained, the sentiment questions quietly regress.

## Definition of done (fine-tuning team)
- [ ] Adapter trained (≥ step 20, stable loss) and exported.
- [ ] `domain-ft` served on node1 `:8001` and listed by the LiteLLM proxy.
- [ ] Direct calls for MHQ058/067/080 return correct sentiment + direction.
- [ ] `training/` has scripts, config, prep notes, logs, base-vs-FT metrics, model card.
- [ ] `model` block of `submission.json` filled (if endpoint reachable).

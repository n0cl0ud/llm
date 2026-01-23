# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Context - Devstral Fine-tuning + RAG Pipeline

## Current Server Setup
- Instance: AWS EC2 g6e.xlarge (GPU L4 24GB VRAM)
- Model: Devstral 2 Small 24B (mistralai/Devstral-Small-2-24B-Instruct-2512)
- Inference: vLLM via Docker
- Dev client: Mistral Vibe CLI
- Exposed port: 11434

## Current vLLM Command (KEEP THIS CONFIG)
```bash
docker run -d \
  --name devstral-vllm \
  --gpus all \
  --ipc=host \
  -p 11434:8000 \
  -v /home/ubuntu/.cache/huggingface:/root/.cache/huggingface \
  --restart unless-stopped \
  vllm/vllm-openai:nightly \
  mistralai/Devstral-Small-2-24B-Instruct-2512 \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 81920 \
  --gpu-memory-utilization 0.95 \
  --kv-cache-dtype fp8 \
  --tool-call-parser mistral \
  --enable-auto-tool-choice \
  --enable-prefix-caching \
  --enable-chunked-prefill
```

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────────────┐
│  Vibe CLI   │────▶│  RAG Proxy  │────▶│  vLLM (fine-tuned)  │
│  (dev)      │     │  :11435     │     │  :11434             │
└─────────────┘     └──────┬──────┘     └─────────────────────┘
                          │
                          ▼
                   ┌─────────────┐
                   │   Qdrant    │  ← Conversation memory
                   │   :6333     │
                   └─────────────┘
```

## Services (docker-compose.yml)

| Service | Port | Description |
|---------|------|-------------|
| `vllm` | 11434 | Base model inference |
| `vllm-lora` | 11434 | Fine-tuned model inference |
| `rag-proxy` | 11435 | RAG proxy (base model) |
| `rag-proxy-lora` | 11435 | RAG proxy (fine-tuned) |
| `qdrant` | 6333 | Vector database |
| `finetune` | - | QLoRA training container |
| `ingest` | - | Index logs into Qdrant |

## Makefile Commands

### Inference
- `make vllm` → Base model (no RAG)
- `make vllm-lora` → Fine-tuned model (no RAG)
- `make rag` → Base model + RAG memory
- `make rag-lora` → Fine-tuned + RAG memory

### RAG
- `make ingest` → Index logs from data/ into Qdrant
- `make ingest-clear` → Clear and re-index
- `make stats` → Show Qdrant stats

### Training
- `make train` → QLoRA fine-tuning with local data
- `make train-hf DATASET=name` → Fine-tune with HuggingFace dataset

### Data
- `make pull-local SRC=/path` → Copy logs to data/
- `make pull-s3` / `make push` → S3 sync

## Data Flow

### For RAG (memory):
```
Vibe logs → data/ → make ingest → Qdrant → RAG proxy enriches prompts
```

### For Fine-tuning (behavior):
```
Vibe logs → data/ → make train → LoRA adapter → vLLM-lora
```

## GPU Constraint
vLLM and training cannot run simultaneously (same 24GB GPU).
Makefile stops vLLM before starting training.

Qdrant + RAG proxy run on CPU, so they can run alongside vLLM.

## Configuration

### QLoRA (fine-tuning)
- Rank: 32
- Alpha: 64
- Target: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj

### RAG Proxy
- Embedding model: `all-MiniLM-L6-v2` (runs on CPU)
- Top-K retrieval: 5
- Min similarity score: 0.3
- Collection: `conversations`

### Vibe CLI
For RAG mode, configure endpoint to RAG proxy port:
```
endpoint: http://server:11435
```
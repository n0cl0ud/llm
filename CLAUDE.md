# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Context - Devstral Fine-tuning Pipeline

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

## Goal
Create a clean Docker setup with:
1. **docker-compose.yml** with 3 services:
   - `vllm`: Base model inference (port 11434)
   - `vllm-lora`: Inference with fine-tuned adapter (port 11434)
   - `finetune`: Unsloth container for QLoRA training

2. **Dockerfile.finetune**: Image with Unsloth + QLoRA optimized for L4 24GB

3. **Scripts**:
   - `train_devstral.py`: QLoRA fine-tuning (rank=32, alpha=64)
   - `s3_sync.py`: Pull/push training data from S3

4. **Makefile** for easy commands:
   - `make vllm` → Start base inference
   - `make vllm-lora` → Start fine-tuned inference
   - `make train` → Stop vLLM + run training
   - `make pull/push` → S3 sync

## Data Flow
Vibe CLI (dev machine) → collect logs from ~/.vibe/logs/ → push to S3 → pull on server → train → LoRA adapter → vLLM with --lora-modules

## GPU Constraint
vLLM and training cannot run simultaneously (same 24GB GPU).
Makefile must stop vLLM before starting training.

## LoRA Config for Fine-tuned vLLM
Add this flag to vLLM command:
--lora-modules devstral-custom=/adapters/devstral-lora
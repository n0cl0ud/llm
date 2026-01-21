# Devstral Fine-tuning Pipeline

QLoRA fine-tuning pipeline for Devstral-Small-2-24B, optimized for NVIDIA L4 GPU (24GB VRAM).

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Vibe CLI      │────▶│   data/         │────▶│   Training      │
│ (dev machine)   │     │ (training data) │     │   (QLoRA)       │
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                         │
                                                         ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Vibe CLI      │◀────│   vLLM + LoRA   │◀────│   adapters/     │
│ (inference)     │     │   (port 11434)  │     │ (fine-tuned)    │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

## Prerequisites

- Docker with NVIDIA GPU support
- NVIDIA L4 24GB (or equivalent)
- HuggingFace token (for Mistral model access)

## Installation

```bash
# Clone the repo
git clone <repo-url> && cd llm

# Create directories and .env file
make init
cp .env.example .env

# Edit .env with your credentials
nano .env
```

## Usage

### Inference (base model)

```bash
make vllm
# API available at http://localhost:11434
```

### Fine-tuning

```bash
# 1. Import training data
make pull-local SRC=~/.vibe/logs    # from local folder
# or
make pull-s3                         # from S3

# 2. Run training (automatically stops vLLM)
make train

# 3. Start vLLM with the adapter
make vllm-lora
```

### Available commands

| Command | Description |
|---------|-------------|
| `make vllm` | Start base inference (port 11434) |
| `make vllm-lora` | Start inference with LoRA adapter |
| `make train` | Run QLoRA fine-tuning |
| `make pull-local SRC=/path` | Copy data from local folder |
| `make pull-s3` | Download data from S3 |
| `make push` | Upload adapter to S3 |
| `make stop` | Stop all containers |
| `make status` | Show container and GPU status |
| `make logs` | View vLLM logs |

## Data format

Training data must be in JSON/JSONL format with the following structure:

```json
{
  "messages": [
    {"role": "system", "content": "You are an assistant..."},
    {"role": "user", "content": "Question"},
    {"role": "assistant", "content": "Answer"}
  ]
}
```

Place `.json` or `.jsonl` files in the `data/` folder.

## Configuration

### Environment variables (.env)

```bash
HF_TOKEN=hf_xxx              # HuggingFace token (required)
HF_CACHE=~/.cache/huggingface

# For S3 (optional)
AWS_ACCESS_KEY_ID=xxx
AWS_SECRET_ACCESS_KEY=xxx
S3_BUCKET=my-bucket
```

### QLoRA parameters

Defined in `scripts/train_devstral.py`:

| Parameter | Value |
|-----------|-------|
| LoRA rank | 32 |
| LoRA alpha | 64 |
| Batch size | 1 |
| Gradient accumulation | 8 |
| Learning rate | 2e-4 |

## GPU constraint

The L4 GPU (24GB) cannot run vLLM and training simultaneously. The Makefile handles this automatically:

- `make train` stops vLLM before starting training
- `make vllm` / `make vllm-lora` stops any existing container

## Project structure

```
.
├── docker-compose.yml      # Services: vllm, vllm-lora, finetune
├── Dockerfile.finetune     # Unsloth + QLoRA image
├── Makefile                # Main commands
├── scripts/
│   ├── train_devstral.py   # Fine-tuning script
│   └── s3_sync.py          # S3 sync
├── data/                   # Training data
└── adapters/               # Generated LoRA adapters
```

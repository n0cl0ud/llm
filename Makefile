.PHONY: vllm vllm-lora train pull pull-s3 pull-local push stop build logs status clean

# Default HuggingFace cache location
export HF_CACHE ?= ~/.cache/huggingface

# Stop all running containers (required before switching modes)
stop:
	docker compose --profile base --profile lora --profile train down

# Start base vLLM inference (port 11434)
vllm: stop
	docker compose --profile base up -d vllm
	@echo "vLLM started on port 11434"

# Start vLLM with LoRA adapter (port 11434)
vllm-lora: stop
	@if [ ! -d "adapters/devstral-lora" ]; then \
		echo "Error: No adapter found at adapters/devstral-lora"; \
		echo "Run 'make train' first to create an adapter"; \
		exit 1; \
	fi
	docker compose --profile lora up -d vllm-lora
	@echo "vLLM with LoRA started on port 11434"

# Build the finetune container
build:
	docker compose --profile train build finetune

# Run training (stops vLLM first - GPU constraint)
train: stop build
	@echo "Starting QLoRA fine-tuning..."
	@echo "GPU will be used exclusively for training"
	docker compose --profile train run --rm finetune
	@echo "Training complete. Run 'make vllm-lora' to use the adapter"

# Pull training data from local folder
# Usage: make pull-local SRC=/path/to/vibe/logs
pull-local:
	@if [ -z "$(SRC)" ]; then \
		echo "Error: SRC required. Usage: make pull-local SRC=/path/to/folder"; \
		exit 1; \
	fi
	@echo "Copying data from $(SRC) to data/"
	cp -r $(SRC)/* data/
	@echo "Data copied. Found $$(find data -type f -name '*.json*' | wc -l) JSON files"

# Pull training data from S3
pull-s3: build
	docker compose --profile train run --rm finetune python s3_sync.py pull

# Alias: pull defaults to local instructions
pull:
	@echo "Choose data source:"
	@echo "  make pull-local SRC=/path/to/folder  - Copy from local folder"
	@echo "  make pull-s3                         - Pull from S3 bucket"
	@echo ""
	@echo "For local testing with Vibe logs:"
	@echo "  make pull-local SRC=~/.vibe/logs"

# Push adapter to S3
push: build
	docker compose --profile train run --rm finetune python s3_sync.py push

# View vLLM logs
logs:
	docker compose --profile base --profile lora logs -f

# Check container status
status:
	@echo "=== Running Containers ==="
	@docker ps --filter "name=devstral" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
	@echo ""
	@echo "=== GPU Status ==="
	@nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null || echo "nvidia-smi not available"

# Clean up containers and images
clean: stop
	docker compose --profile base --profile lora --profile train down --rmi local --volumes
	@echo "Cleaned up containers and local images"

# Create required directories
init:
	mkdir -p data adapters
	@echo "Created data/ and adapters/ directories"

# Help
help:
	@echo "Devstral Fine-tuning Pipeline"
	@echo ""
	@echo "Inference:"
	@echo "  make vllm       - Start base model inference (port 11434)"
	@echo "  make vllm-lora  - Start inference with fine-tuned adapter"
	@echo "  make stop       - Stop all containers"
	@echo "  make logs       - View container logs"
	@echo "  make status     - Show container and GPU status"
	@echo ""
	@echo "Training:"
	@echo "  make train      - Run QLoRA fine-tuning (stops vLLM first)"
	@echo "  make build      - Build finetune container"
	@echo ""
	@echo "Data Sync:"
	@echo "  make pull-local SRC=/path  - Copy training data from local folder"
	@echo "  make pull-s3               - Pull training data from S3"
	@echo "  make push                  - Push adapter to S3"
	@echo ""
	@echo "Setup:"
	@echo "  make init       - Create required directories"
	@echo "  make clean      - Remove containers and images"
	@echo ""
	@echo "Environment variables:"
	@echo "  HF_CACHE        - HuggingFace cache dir (default: ~/.cache/huggingface)"
	@echo "  HF_TOKEN        - HuggingFace API token"
	@echo "  S3_BUCKET       - S3 bucket for data sync"
	@echo "  AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY - AWS credentials"

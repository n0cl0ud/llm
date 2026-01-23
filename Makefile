.PHONY: vllm vllm-lora rag rag-lora ingest train train-hf pull pull-s3 pull-local push stop stop-rag build build-rag logs status stats clean

# Default HuggingFace cache location
export HF_CACHE ?= ~/.cache/huggingface

# Stop all running containers (required before switching modes)
stop:
	docker compose --profile base --profile lora --profile train --profile rag --profile rag-lora --profile ingest down

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

# =============================================================================
# RAG Mode - vLLM + Qdrant + RAG Proxy
# =============================================================================

# Start RAG mode with base model (proxy on 11435, vLLM on 11434)
rag: stop build-rag
	docker compose --profile rag up -d qdrant vllm rag-proxy
	@echo ""
	@echo "RAG mode started:"
	@echo "  - vLLM (direct):    http://localhost:11434"
	@echo "  - RAG proxy:        http://localhost:11435  <- Use this in Vibe"
	@echo "  - Qdrant dashboard: http://localhost:6333/dashboard"

# Start RAG mode with fine-tuned model
rag-lora: stop build-rag
	@if [ ! -d "adapters/devstral-lora" ]; then \
		echo "Error: No adapter found at adapters/devstral-lora"; \
		echo "Run 'make train' first to create an adapter"; \
		exit 1; \
	fi
	docker compose --profile rag-lora up -d qdrant vllm-lora rag-proxy-lora
	@echo ""
	@echo "RAG mode (LoRA) started:"
	@echo "  - vLLM (direct):    http://localhost:11434"
	@echo "  - RAG proxy:        http://localhost:11435  <- Use this in Vibe"
	@echo "  - Qdrant dashboard: http://localhost:6333/dashboard"

# Ingest conversation logs into Qdrant
ingest: build-rag
	@if [ ! -d "data" ] || [ -z "$$(ls -A data 2>/dev/null)" ]; then \
		echo "Error: No data found in data/ directory"; \
		echo "Run 'make pull-local SRC=/path/to/logs' first"; \
		exit 1; \
	fi
	docker compose --profile rag up -d qdrant
	@echo "Waiting for Qdrant to be ready..."
	@sleep 3
	docker compose --profile ingest run --rm ingest
	@echo ""
	@echo "Ingestion complete! Run 'make stats' to see index size"

# Ingest with --clear flag (reindex everything)
ingest-clear: build-rag
	docker compose --profile rag up -d qdrant
	@sleep 3
	docker compose --profile ingest run --rm ingest python ingest.py --data-dir /data --clear
	@echo "Re-indexed all documents"

# Stop RAG services (keeps Qdrant data)
stop-rag:
	docker compose --profile rag --profile rag-lora down
	@echo "RAG services stopped (Qdrant data preserved in qdrant_data/)"

# Build RAG proxy image
build-rag:
	docker compose --profile rag build rag-proxy

# Show RAG statistics
stats:
	@curl -s http://localhost:6333/collections/conversations 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(f\"Collection: conversations\nVectors: {d['result']['points_count']}\nStatus: {d['result']['status']}\")" 2>/dev/null || echo "Qdrant not running. Start with 'make rag'"

# =============================================================================
# Training
# =============================================================================

# Build the finetune container
build:
	docker compose --profile train build finetune

# Run training with local data (stops vLLM first - GPU constraint)
train: stop build
	@echo "Starting QLoRA fine-tuning with local data..."
	@echo "GPU will be used exclusively for training"
	docker compose --profile train run --rm finetune
	@echo "Training complete. Run 'make vllm-lora' to use the adapter"

# Run training with HuggingFace dataset
# Usage: make train-hf DATASET=username/dataset-name
train-hf: stop build
	@if [ -z "$(DATASET)" ]; then \
		echo "Error: DATASET required. Usage: make train-hf DATASET=username/dataset"; \
		echo ""; \
		echo "Examples:"; \
		echo "  make train-hf DATASET=mlabonne/FineTome-100k"; \
		echo "  make train-hf DATASET=OpenAssistant/oasst1 SPLIT=train"; \
		exit 1; \
	fi
	@echo "Starting QLoRA fine-tuning with HuggingFace dataset: $(DATASET)"
	docker compose --profile train run --rm finetune python train_devstral.py \
		--dataset $(DATASET) \
		$(if $(SPLIT),--split $(SPLIT)) \
		$(if $(TEXT_FIELD),--text-field $(TEXT_FIELD)) \
		$(if $(MAX_SAMPLES),--max-samples $(MAX_SAMPLES))
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
	docker compose --profile base --profile lora --profile train --profile rag --profile rag-lora down --rmi local --volumes
	rm -rf qdrant_data
	@echo "Cleaned up containers, images, and Qdrant data"

# Create required directories
init:
	mkdir -p data adapters qdrant_data
	@echo "Created data/, adapters/, and qdrant_data/ directories"

# Help
help:
	@echo "Devstral Fine-tuning + RAG Pipeline"
	@echo ""
	@echo "Inference (without RAG):"
	@echo "  make vllm       - Start base model (port 11434)"
	@echo "  make vllm-lora  - Start fine-tuned model (port 11434)"
	@echo ""
	@echo "RAG Mode (with memory):"
	@echo "  make rag        - Start base model + RAG (proxy on 11435)"
	@echo "  make rag-lora   - Start fine-tuned + RAG (proxy on 11435)"
	@echo "  make ingest     - Index logs from data/ into Qdrant"
	@echo "  make ingest-clear - Re-index (clear + ingest)"
	@echo "  make stats      - Show Qdrant collection stats"
	@echo ""
	@echo "Training:"
	@echo "  make train                        - Train with local data"
	@echo "  make train-hf DATASET=user/name   - Train with HuggingFace dataset"
	@echo "  make build                        - Build finetune container"
	@echo ""
	@echo "Data Sync:"
	@echo "  make pull-local SRC=/path  - Copy training data from local folder"
	@echo "  make pull-s3               - Pull training data from S3"
	@echo "  make push                  - Push adapter to S3"
	@echo ""
	@echo "Management:"
	@echo "  make stop       - Stop all containers"
	@echo "  make logs       - View container logs"
	@echo "  make status     - Show container and GPU status"
	@echo "  make init       - Create required directories"
	@echo "  make clean      - Remove containers and images"
	@echo ""
	@echo "Environment variables:"
	@echo "  HF_CACHE        - HuggingFace cache dir (default: ~/.cache/huggingface)"
	@echo "  HF_TOKEN        - HuggingFace API token"
	@echo "  S3_BUCKET       - S3 bucket for data sync"
	@echo "  AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY - AWS credentials"
	@echo ""
	@echo "Vibe CLI config (for RAG mode):"
	@echo "  endpoint: http://server:11435  (use RAG proxy port)"

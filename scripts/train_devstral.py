#!/usr/bin/env python3
"""
QLoRA fine-tuning script for Devstral-Small-2-24B using Unsloth.
Optimized for NVIDIA L4 24GB GPU.
"""

import os
import json
import argparse
from pathlib import Path

from unsloth import FastLanguageModel
from datasets import Dataset, load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# Model configuration
MODEL_NAME = "mistralai/Devstral-Small-2-24B-Instruct-2512"
MAX_SEQ_LENGTH = 8192  # Reduced for training memory constraints

# QLoRA configuration (as specified in CLAUDE.md)
LORA_RANK = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.0

# Training defaults optimized for L4 24GB
DEFAULT_BATCH_SIZE = 1
DEFAULT_GRAD_ACCUM = 8
DEFAULT_EPOCHS = 3
DEFAULT_LR = 2e-4


def load_local_data(data_dir: str) -> list[dict]:
    """Load training data from local JSON/JSONL files."""
    data_path = Path(data_dir)
    conversations = []

    # Look for JSONL files
    for jsonl_file in data_path.glob("*.jsonl"):
        with open(jsonl_file, "r") as f:
            for line in f:
                if line.strip():
                    conversations.append(json.loads(line))

    # Look for JSON files
    for json_file in data_path.glob("*.json"):
        with open(json_file, "r") as f:
            data = json.load(f)
            if isinstance(data, list):
                conversations.extend(data)
            else:
                conversations.append(data)

    return conversations


def load_hf_dataset(dataset_name: str, split: str = "train", text_field: str = None):
    """Load dataset from HuggingFace Hub."""
    print(f"Loading dataset from HuggingFace: {dataset_name} (split: {split})")
    dataset = load_dataset(dataset_name, split=split)
    return dataset, text_field


def format_conversation(example: dict) -> dict:
    """Format conversation for Mistral instruction format."""
    messages = example.get("messages", [])

    # Build conversation text in Mistral format
    text_parts = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "user":
            text_parts.append(f"[INST] {content} [/INST]")
        elif role == "assistant":
            text_parts.append(f"{content}</s>")
        elif role == "system":
            text_parts.insert(0, f"[INST] {content}\n")

    return {"text": "".join(text_parts)}


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Devstral with QLoRA")

    # Data source (mutually exclusive)
    data_group = parser.add_mutually_exclusive_group()
    data_group.add_argument("--data-dir", default=None, help="Local training data directory")
    data_group.add_argument("--dataset", default=None, help="HuggingFace dataset name (e.g., 'username/dataset')")

    # HuggingFace dataset options
    parser.add_argument("--split", default="train", help="Dataset split to use (default: train)")
    parser.add_argument("--text-field", default=None, help="Field containing text (if not using 'messages' format)")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit number of training samples")

    # Training config
    parser.add_argument("--output-dir", default="/adapters/devstral-lora", help="Output adapter directory")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Per-device batch size")
    parser.add_argument("--grad-accum", type=int, default=DEFAULT_GRAD_ACCUM, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR, help="Learning rate")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    args = parser.parse_args()

    # Default to local data if no source specified
    if args.data_dir is None and args.dataset is None:
        args.data_dir = "/data"

    print(f"Loading model: {MODEL_NAME}")
    print(f"QLoRA config: rank={LORA_RANK}, alpha={LORA_ALPHA}")

    # Load model with Unsloth optimizations
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,  # Auto-detect
        load_in_4bit=True,  # QLoRA 4-bit quantization
    )

    # Add LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    # Load dataset from HuggingFace or local files
    if args.dataset:
        # Load from HuggingFace Hub
        dataset = load_dataset(args.dataset, split=args.split)
        print(f"Loaded {len(dataset)} samples from HuggingFace: {args.dataset}")

        # Determine text field
        text_field = args.text_field
        if text_field is None:
            # Auto-detect: check for common field names
            if "messages" in dataset.column_names:
                # Format conversations to text
                dataset = dataset.map(format_conversation)
                text_field = "text"
            elif "text" in dataset.column_names:
                text_field = "text"
            elif "content" in dataset.column_names:
                text_field = "content"
            else:
                print(f"Available fields: {dataset.column_names}")
                print("Please specify --text-field")
                return
        print(f"Using text field: {text_field}")

    else:
        # Load from local files
        print(f"Loading training data from: {args.data_dir}")
        raw_data = load_local_data(args.data_dir)

        if not raw_data:
            print("No training data found. Please run 'make pull-local' or 'make pull-s3' first.")
            return

        print(f"Loaded {len(raw_data)} conversations")

        # Format dataset
        formatted_data = [format_conversation(ex) for ex in raw_data]
        dataset = Dataset.from_list(formatted_data)
        text_field = "text"

    # Limit samples if requested
    if args.max_samples and len(dataset) > args.max_samples:
        dataset = dataset.select(range(args.max_samples))
        print(f"Limited to {args.max_samples} samples")

    # Training arguments optimized for L4 24GB
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        bf16=True,
        optim="adamw_8bit",
        seed=42,
        report_to="none",
    )

    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field=text_field,
        max_seq_length=MAX_SEQ_LENGTH,
        args=training_args,
    )

    # Train
    print("Starting training...")
    trainer.train(resume_from_checkpoint=args.resume)

    # Save the adapter
    print(f"Saving adapter to: {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("Training complete!")


if __name__ == "__main__":
    main()

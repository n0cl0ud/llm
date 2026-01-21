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
from datasets import Dataset
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


def load_vibe_logs(data_dir: str) -> list[dict]:
    """Load training data from Vibe CLI logs."""
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
    parser.add_argument("--data-dir", default="/data", help="Training data directory")
    parser.add_argument("--output-dir", default="/adapters/devstral-lora", help="Output adapter directory")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Per-device batch size")
    parser.add_argument("--grad-accum", type=int, default=DEFAULT_GRAD_ACCUM, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR, help="Learning rate")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    args = parser.parse_args()

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

    # Load and prepare dataset
    print(f"Loading training data from: {args.data_dir}")
    raw_data = load_vibe_logs(args.data_dir)

    if not raw_data:
        print("No training data found. Please run 'make pull' first.")
        return

    print(f"Loaded {len(raw_data)} conversations")

    # Format dataset
    formatted_data = [format_conversation(ex) for ex in raw_data]
    dataset = Dataset.from_list(formatted_data)

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
        dataset_text_field="text",
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

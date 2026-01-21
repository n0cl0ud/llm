#!/bin/bash
set -e

# Login to Hugging Face if token provided
if [ -n "$HF_TOKEN" ]; then
    echo "Logging in to Hugging Face..."
    huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential
fi

# Execute the command
exec "$@"

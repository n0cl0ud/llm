#!/usr/bin/env python3
"""
Ingest script - Index Vibe CLI conversation logs into Qdrant for RAG retrieval.
Supports both local files and S3.
"""

import os
import json
import hashlib
import argparse
from pathlib import Path
from datetime import datetime

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

# Configuration
QDRANT_URL = os.environ.get("QDRANT_URL", "http://qdrant:6333")
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "conversations")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
BATCH_SIZE = 100


def generate_id(content: str, timestamp: str = "") -> str:
    """Generate a deterministic ID for deduplication."""
    data = f"{content}:{timestamp}"
    return hashlib.md5(data.encode()).hexdigest()


def load_vibe_logs(data_dir: str) -> list[dict]:
    """Load conversation logs from Vibe CLI format."""
    data_path = Path(data_dir)
    documents = []

    # Process JSONL files
    for jsonl_file in data_path.glob("**/*.jsonl"):
        print(f"Processing: {jsonl_file}")
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                    docs = parse_log_entry(entry, str(jsonl_file))
                    documents.extend(docs)
                except json.JSONDecodeError as e:
                    print(f"  Skipping line {line_num}: {e}")

    # Process JSON files
    for json_file in data_path.glob("**/*.json"):
        print(f"Processing: {json_file}")
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, list):
                for entry in data:
                    docs = parse_log_entry(entry, str(json_file))
                    documents.extend(docs)
            else:
                docs = parse_log_entry(data, str(json_file))
                documents.extend(docs)
        except (json.JSONDecodeError, Exception) as e:
            print(f"  Error processing {json_file}: {e}")

    return documents


def parse_log_entry(entry: dict, source: str) -> list[dict]:
    """Parse a single log entry into indexable documents."""
    documents = []

    # Handle messages array format
    messages = entry.get("messages", [])
    if messages:
        timestamp = entry.get("timestamp", entry.get("created_at", ""))
        session_id = entry.get("session_id", entry.get("id", ""))

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if not content or role == "system":
                continue

            # Skip very short messages
            if len(content.strip()) < 10:
                continue

            documents.append({
                "content": content,
                "role": role,
                "timestamp": timestamp,
                "session_id": session_id,
                "source": source,
            })

    # Handle flat format (single message)
    elif "content" in entry or "text" in entry:
        content = entry.get("content") or entry.get("text", "")
        role = entry.get("role", "unknown")
        timestamp = entry.get("timestamp", entry.get("created_at", ""))

        if content and len(content.strip()) >= 10:
            documents.append({
                "content": content,
                "role": role,
                "timestamp": timestamp,
                "source": source,
            })

    # Handle conversation format
    elif "conversation" in entry:
        conversation = entry.get("conversation", [])
        timestamp = entry.get("timestamp", "")

        for turn in conversation:
            if isinstance(turn, dict):
                content = turn.get("content") or turn.get("text", "")
                role = turn.get("role", "unknown")
            elif isinstance(turn, str):
                content = turn
                role = "unknown"
            else:
                continue

            if content and len(content.strip()) >= 10:
                documents.append({
                    "content": content,
                    "role": role,
                    "timestamp": timestamp,
                    "source": source,
                })

    return documents


def chunk_text(text: str, max_length: int = 512, overlap: int = 50) -> list[str]:
    """Split long text into overlapping chunks."""
    if len(text) <= max_length:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + max_length
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap

    return chunks


def ingest_documents(
    documents: list[dict],
    qdrant: QdrantClient,
    embedder: SentenceTransformer,
    collection_name: str,
):
    """Ingest documents into Qdrant."""
    print(f"\nIngesting {len(documents)} documents into collection '{collection_name}'")

    # Ensure collection exists
    collections = [c.name for c in qdrant.get_collections().collections]
    if collection_name not in collections:
        print(f"Creating collection: {collection_name}")
        qdrant.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=embedder.get_sentence_embedding_dimension(),
                distance=Distance.COSINE,
            ),
        )

    # Process in batches
    points = []
    seen_ids = set()

    for doc in documents:
        content = doc["content"]

        # Chunk long content
        chunks = chunk_text(content)

        for i, chunk in enumerate(chunks):
            doc_id = generate_id(chunk, doc.get("timestamp", ""))

            # Skip duplicates
            if doc_id in seen_ids:
                continue
            seen_ids.add(doc_id)

            # Generate embedding
            embedding = embedder.encode(chunk).tolist()

            points.append(PointStruct(
                id=doc_id,
                vector=embedding,
                payload={
                    "content": chunk,
                    "role": doc.get("role", ""),
                    "timestamp": doc.get("timestamp", ""),
                    "session_id": doc.get("session_id", ""),
                    "source": doc.get("source", ""),
                    "chunk_index": i,
                    "indexed_at": datetime.utcnow().isoformat(),
                },
            ))

            # Batch upsert
            if len(points) >= BATCH_SIZE:
                qdrant.upsert(collection_name=collection_name, points=points)
                print(f"  Indexed {len(points)} points...")
                points = []

    # Final batch
    if points:
        qdrant.upsert(collection_name=collection_name, points=points)
        print(f"  Indexed {len(points)} points...")

    # Stats
    info = qdrant.get_collection(collection_name)
    print(f"\nCollection '{collection_name}' now has {info.points_count} points")


def clear_collection(qdrant: QdrantClient, collection_name: str):
    """Delete and recreate collection."""
    try:
        qdrant.delete_collection(collection_name)
        print(f"Deleted collection: {collection_name}")
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description="Ingest conversation logs into Qdrant")
    parser.add_argument("--data-dir", default="/data", help="Directory containing log files")
    parser.add_argument("--qdrant-url", default=QDRANT_URL, help="Qdrant server URL")
    parser.add_argument("--collection", default=COLLECTION_NAME, help="Collection name")
    parser.add_argument("--model", default=EMBEDDING_MODEL, help="Embedding model")
    parser.add_argument("--clear", action="store_true", help="Clear collection before ingesting")
    args = parser.parse_args()

    print(f"Qdrant URL: {args.qdrant_url}")
    print(f"Collection: {args.collection}")
    print(f"Embedding model: {args.model}")
    print(f"Data directory: {args.data_dir}")

    # Initialize clients
    qdrant = QdrantClient(url=args.qdrant_url)
    print("\nLoading embedding model...")
    embedder = SentenceTransformer(args.model)

    # Clear if requested
    if args.clear:
        clear_collection(qdrant, args.collection)

    # Load documents
    print(f"\nLoading logs from {args.data_dir}...")
    documents = load_vibe_logs(args.data_dir)

    if not documents:
        print("No documents found to ingest.")
        print("Make sure your data directory contains .json or .jsonl files.")
        return

    print(f"Found {len(documents)} messages to index")

    # Ingest
    ingest_documents(documents, qdrant, embedder, args.collection)

    print("\nIngestion complete!")


if __name__ == "__main__":
    main()

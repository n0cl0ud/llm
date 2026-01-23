#!/usr/bin/env python3
"""
RAG Proxy - OpenAI-compatible API that enriches prompts with context from Qdrant.
Sits between Vibe CLI and vLLM, transparently adding relevant conversation history.
"""

import os
import httpx
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer

# Configuration
VLLM_URL = os.environ.get("VLLM_URL", "http://vllm:8000")
QDRANT_URL = os.environ.get("QDRANT_URL", "http://qdrant:6333")
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "conversations")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
TOP_K = int(os.environ.get("RAG_TOP_K", "5"))
MIN_SCORE = float(os.environ.get("RAG_MIN_SCORE", "0.3"))

# Global clients
qdrant: QdrantClient = None
embedder: SentenceTransformer = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize clients on startup."""
    global qdrant, embedder

    print(f"Connecting to Qdrant at {QDRANT_URL}")
    qdrant = QdrantClient(url=QDRANT_URL)

    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    embedder = SentenceTransformer(EMBEDDING_MODEL)

    # Ensure collection exists
    collections = [c.name for c in qdrant.get_collections().collections]
    if COLLECTION_NAME not in collections:
        print(f"Creating collection: {COLLECTION_NAME}")
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=embedder.get_sentence_embedding_dimension(),
                distance=Distance.COSINE,
            ),
        )

    yield

    # Cleanup
    qdrant.close()


app = FastAPI(title="RAG Proxy", lifespan=lifespan)


def retrieve_context(query: str, limit: int = TOP_K) -> list[dict]:
    """Retrieve relevant context from Qdrant."""
    if not qdrant or not embedder:
        return []

    try:
        query_vector = embedder.encode(query).tolist()

        results = qdrant.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=limit,
            score_threshold=MIN_SCORE,
        )

        contexts = []
        for hit in results:
            contexts.append({
                "content": hit.payload.get("content", ""),
                "role": hit.payload.get("role", ""),
                "timestamp": hit.payload.get("timestamp", ""),
                "score": hit.score,
            })

        return contexts

    except Exception as e:
        print(f"RAG retrieval error: {e}")
        return []


def build_rag_context(contexts: list[dict]) -> str:
    """Format retrieved contexts into a system message."""
    if not contexts:
        return ""

    parts = ["<relevant_history>"]
    for ctx in contexts:
        role = ctx.get("role", "unknown")
        content = ctx.get("content", "")
        parts.append(f"[{role}]: {content}")
    parts.append("</relevant_history>")

    return "\n".join(parts)


def enrich_messages(messages: list[dict], rag_context: str) -> list[dict]:
    """Inject RAG context into the messages."""
    if not rag_context or not messages:
        return messages

    enriched = messages.copy()

    # Find or create system message
    system_idx = None
    for i, msg in enumerate(enriched):
        if msg.get("role") == "system":
            system_idx = i
            break

    rag_instruction = (
        "\n\nYou have access to relevant conversation history below. "
        "Use it if helpful, but don't mention that you're using retrieved context.\n\n"
        f"{rag_context}"
    )

    if system_idx is not None:
        enriched[system_idx] = {
            **enriched[system_idx],
            "content": enriched[system_idx].get("content", "") + rag_instruction,
        }
    else:
        enriched.insert(0, {
            "role": "system",
            "content": f"You are a helpful assistant.{rag_instruction}",
        })

    return enriched


async def proxy_stream(request_data: dict) -> AsyncGenerator[bytes, None]:
    """Stream response from vLLM."""
    async with httpx.AsyncClient(timeout=300.0) as client:
        async with client.stream(
            "POST",
            f"{VLLM_URL}/v1/chat/completions",
            json=request_data,
        ) as response:
            async for chunk in response.aiter_bytes():
                yield chunk


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """OpenAI-compatible chat completions endpoint with RAG."""
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    messages = body.get("messages", [])

    # Extract the last user message for retrieval
    user_query = None
    for msg in reversed(messages):
        if msg.get("role") == "user":
            user_query = msg.get("content", "")
            break

    # Retrieve relevant context
    if user_query:
        contexts = retrieve_context(user_query)
        if contexts:
            rag_context = build_rag_context(contexts)
            messages = enrich_messages(messages, rag_context)
            body["messages"] = messages
            print(f"RAG: Found {len(contexts)} relevant contexts for query")

    # Check if streaming
    if body.get("stream", False):
        return StreamingResponse(
            proxy_stream(body),
            media_type="text/event-stream",
        )

    # Non-streaming request
    async with httpx.AsyncClient(timeout=300.0) as client:
        response = await client.post(
            f"{VLLM_URL}/v1/chat/completions",
            json=body,
        )
        return JSONResponse(
            content=response.json(),
            status_code=response.status_code,
        )


@app.get("/v1/models")
async def list_models():
    """Proxy models endpoint to vLLM."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(f"{VLLM_URL}/v1/models")
        return JSONResponse(content=response.json(), status_code=response.status_code)


@app.get("/health")
async def health():
    """Health check endpoint."""
    try:
        # Check Qdrant
        qdrant.get_collections()
        qdrant_ok = True
    except Exception:
        qdrant_ok = False

    # Check vLLM
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{VLLM_URL}/health")
            vllm_ok = response.status_code == 200
    except Exception:
        vllm_ok = False

    return {
        "status": "healthy" if (qdrant_ok and vllm_ok) else "degraded",
        "qdrant": "ok" if qdrant_ok else "error",
        "vllm": "ok" if vllm_ok else "error",
    }


@app.get("/stats")
async def stats():
    """Get RAG statistics."""
    try:
        info = qdrant.get_collection(COLLECTION_NAME)
        return {
            "collection": COLLECTION_NAME,
            "vectors_count": info.vectors_count,
            "points_count": info.points_count,
        }
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

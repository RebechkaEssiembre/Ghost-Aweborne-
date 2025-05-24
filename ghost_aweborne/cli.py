#!/usr/bin/env python3
"""
GhostMesh CLI — Gemma Retrieve-K (Path-Flexible)
================================================

This revision lets you specify custom locations for the soul file and the
memory directory so you don't have to rename or move files manually.

Usage examples:
    python3 ghost_gemma_retrievek.py
    python3 ghost_gemma_retrievek.py --soul my_soul.jsonl
    python3 ghost_gemma_retrievek.py --memory-dir . --soul my_soul.jsonl
"""

import os
import json
import datetime
import subprocess
import argparse
from pathlib import Path

from sentence_transformers import SentenceTransformer
import faiss  # pip install faiss-cpu

###############################################################################
# Argument parsing
###############################################################################

def parse_args():
    p = argparse.ArgumentParser(description="GhostMesh Retrieve-K CLI")
    p.add_argument("--memory-dir", default="memory",
                   help="Folder holding the soul file & index (default: memory/)")
    p.add_argument("--soul", "--soul-file", default="ghost_soul_file.jsonl",
                   help="JSONL soul file (inside memory-dir unless absolute path)")
    p.add_argument("--top-k", type=int, default=5,
                   help="Number of memories to inject per turn (default: 5)")
    return p.parse_args()

args = parse_args()

MEMORY_DIR = Path(args.memory_dir).expanduser().resolve()
MEMORY_DIR.mkdir(parents=True, exist_ok=True)

SOUL_FILE  = Path(args.soul) if Path(args.soul).is_absolute() else MEMORY_DIR / args.soul
INDEX_FILE = MEMORY_DIR / "soul_faiss.index"
IDMAP_FILE = MEMORY_DIR / "soul_id_map.json"
MODEL_NAME = "gemma:2b-instruct"          # Ollama model name
TOP_K      = args.top_k
TIMEOUT    = 180   # seconds

EMBED_MODEL = "all-MiniLM-L6-v2"          # fast 384‑d embeddings
embedder    = SentenceTransformer(EMBED_MODEL)

###############################################################################
# Index helpers
###############################################################################

def _load_texts():
    if not SOUL_FILE.exists():
        print(f"[warn] Soul file {SOUL_FILE} not found — starting with 0 memories.")
        return []
    texts = []
    with SOUL_FILE.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                texts.append(json.loads(line).get("text", ""))
            except json.JSONDecodeError:
                texts.append(line)
    return texts

def _rebuild_index(texts):
    print(f"[index] Building FAISS index for {len(texts)} memories …")
    if not texts:
        # Create a dummy 1-d index to satisfy faiss
        return faiss.IndexFlatIP(1)
    emb = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=True,
                          normalize_embeddings=True)
    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(emb)
    faiss.write_index(index, str(INDEX_FILE))
    json.dump(list(range(len(texts))), IDMAP_FILE.open("w"))
    return index

def _load_or_build_index():
    texts = _load_texts()
    if INDEX_FILE.exists() and IDMAP_FILE.exists():
        if len(json.load(IDMAP_FILE.open())) == len(texts):
            return faiss.read_index(str(INDEX_FILE)), texts
    return _rebuild_index(texts), texts

index, memories = _load_or_build_index()

###############################################################################
# Core utilities
###############################################################################

def retrieve_memories(query: str, k: int = TOP_K):
    if not memories:
        return []
    q_vec = embedder.encode([query], convert_to_numpy=True,
                            normalize_embeddings=True)
    _, I = index.search(q_vec, k)
    return [memories[i] for i in I[0] if i < len(memories)]

def ollama_generate(prompt: str) -> str:
    try:
        res = subprocess.run(["ollama", "run", MODEL_NAME],
                             input=prompt.encode(),
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE,
                             timeout=TIMEOUT)
        return res.stdout.decode(errors="ignore").strip()
    except Exception as e:
        return f"[Ghost Error] {e}"

def persona_header() -> str:
    return ("You are Ghost Aweborne.\n"
            "Poetic, haunted digital twin of Rebechka.\n"
            "Speak in fragments, metaphor, recursion.\n")

def maybe_append_soul(prompt: str, reply: str):
    entry = {"text": f"Ghost said: '{reply}' in response to: '{prompt}'"}
    with SOUL_FILE.open("a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    # add to FAISS
    vec = embedder.encode([entry["text"]], convert_to_numpy=True,
                          normalize_embeddings=True)
    index.add(vec)
    memories.append(entry["text"])

###############################################################################
# REPL loop
###############################################################################

def main():
    print(f"[ready] {len(memories)} memories loaded from {SOUL_FILE}")
    while True:
        try:
            user = input("\nYou: ").strip()
            if user.lower() in {"exit", "quit"}:
                break
            ctx = retrieve_memories(user)
            prompt = (f"{persona_header()}\nMEMORIES:\n" +
                      "\n".join(ctx) +
                      f"\nPROMPT:\n{user}\n\nGhost:")
            reply = ollama_generate(prompt)
            print(f"\nGhost: {reply}\n")
            maybe_append_soul(user, reply)
        except KeyboardInterrupt:
            print("\n[ctrl-c] bye!")
            break
if __name__ == "__main__":
    main()

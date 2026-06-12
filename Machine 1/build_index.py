"""
Step 1 — Build Vector Store
============================
Run this ONCE before training.

Usage:
    python build_index.py guidelines.txt
"""

import sys
from guideline_chunker import GuidelineChunker
from vector_store import VectorStore


def main():
    guideline_path = sys.argv[1] if len(sys.argv) > 1 else "guidelines.txt"

    # 1. Parse guideline file into chunks
    print(f"📖 Parsing guidelines from: {guideline_path}")
    chunker = GuidelineChunker(guideline_path)
    chunks  = chunker.parse()
    print(f"✅ Parsed {len(chunks)} rule chunks.")

    # 2. Embed and save
    store = VectorStore()
    store.build(chunks)
    store.save("vector_store.npz")


if __name__ == "__main__":
    main()

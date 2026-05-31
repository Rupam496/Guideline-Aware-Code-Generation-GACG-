import json
import gc
import numpy as np
import torch

from typing import List
from sentence_transformers import SentenceTransformer
from guideline_chunker import RuleChunk


class VectorStore:

    def __init__(self, model_name="jinaai/jina-embeddings-v2-base-code"):
        self.model_name = model_name
        self.model = None
        self.chunks: List[RuleChunk] = []
        self.embeddings = None

    # -----------------------------------
    # Build vector store ON GPU (one time)
    # -----------------------------------
    def build(self, chunks: List[RuleChunk]):
        self.chunks = chunks
        texts = [c.as_flat_text() for c in chunks]

        print("🔄 Loading embedder on cuda for indexing...")
        model = SentenceTransformer(self.model_name, device="cuda")

        self.embeddings = model.encode(
            texts,
            batch_size=2,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        print(f"✅ Indexed {len(chunks)} chunks.")

        # Free GPU after build
        del model
        gc.collect()
        torch.cuda.empty_cache()

    # -----------------------------------
    # Save
    # -----------------------------------
    def save(self, path="vector_store.npz"):
        np.savez(path, embeddings=self.embeddings)
        with open(path.replace(".npz", "_chunks.json"), "w") as f:
            json.dump([c.to_dict() for c in self.chunks], f, indent=2)
        print(f"💾 Saved to {path}")

    # -----------------------------------
    # Load — also loads embedder on CPU once
    # -----------------------------------
    def load(self, path="vector_store.npz"):
        data = np.load(path)
        self.embeddings = data["embeddings"]

        with open(path.replace(".npz", "_chunks.json")) as f:
            raw = json.load(f)

        self.chunks = [RuleChunk(**r) for r in raw]

        print("🔄 Loading embedder on CPU...")
        self.model = SentenceTransformer(self.model_name, device="cpu")

        print(f"✅ Loaded {len(self.chunks)} chunks")

    # -----------------------------------
    # Retrieve ON CPU — no load/unload
    # -----------------------------------
    def retrieve(self, query: str, top_k: int = 8) -> List[RuleChunk]:
        q = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        scores = q @ self.embeddings.T
        idx = np.argsort(scores[0])[::-1][:top_k]
        return [self.chunks[i] for i in idx]
"""Shared embedding helper. EMBED_MODEL env var chooses the model.
  BAAI/bge-small-en-v1.5        (default, English)
  intfloat/multilingual-e5-base (English + Marathi/Hindi queries)
  hash                          (offline smoke test only, not real semantics)
"""
import os, re, hashlib
import numpy as np

EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder:
    def __init__(self, name: str = EMBED_MODEL):
        self.name = name
        self.model = None
        if name != "hash":
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(name)

    def _hash(self, texts):
        out = np.zeros((len(texts), 256), dtype="float32")
        for i, t in enumerate(texts):
            for w in re.findall(r"\w+", t.lower()):
                out[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % 256] += 1
        out /= np.linalg.norm(out, axis=1, keepdims=True) + 1e-9
        return out

    def embed_docs(self, texts):
        if self.model is None:
            return self._hash(texts).tolist()
        if "e5" in self.name:
            texts = ["passage: " + t for t in texts]
        return self.model.encode(texts, normalize_embeddings=True, batch_size=32).tolist()

    def embed_query(self, text):
        if self.model is None:
            return self._hash([text])[0].tolist()
        if "e5" in self.name:
            text = "query: " + text
        elif "bge" in self.name:
            text = BGE_QUERY_PREFIX + text
        return self.model.encode([text], normalize_embeddings=True)[0].tolist()

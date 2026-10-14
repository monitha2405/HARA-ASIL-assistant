"""RAG knowledge base: approved HARA library stored in ChromaDB (local, persistent).
Embeddings: Chroma's default ONNX MiniLM model (small, free, runs on CPU).
If that model cannot be downloaded (offline machine) we fall back to a hashing embedder."""
import pandas as pd, numpy as np
import chromadb
from sklearn.feature_extraction.text import HashingVectorizer
from .data_gen import compose

PATH = "models/chroma"
_hash = HashingVectorizer(n_features=384, alternate_sign=False, norm="l2", ngram_range=(1, 2))

class Embedder(chromadb.EmbeddingFunction):
    """Tries the MiniLM sentence embedder first, otherwise hashing."""
    def __init__(self):
        self.name_used = "hashing-fallback"
        try:
            from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
            self._main = DefaultEmbeddingFunction(); self._main(["test"]); self.name_used = "MiniLM-L6-v2 (ONNX)"
        except Exception:
            self._main = None
    def __call__(self, input):
        if self._main is not None:
            return [list(map(float, v)) for v in self._main(input)]
        return _hash.transform(input).toarray().tolist()
    @staticmethod
    def name(): return "hara_embedder"
    def get_config(self): return {}
    @staticmethod
    def build_from_config(cfg): return Embedder()

def _client(): return chromadb.PersistentClient(path=PATH)

def build_library(csv="data/hara_synthetic.csv"):
    """Build the approved-hazard library: one entry per distinct hazard text with its consensus S/E/C/ASIL."""
    df = pd.read_csv(csv)
    # group identical hazard texts and keep the most common (consensus) S/E/C label
    g = df.groupby("text").agg(S=("S", lambda x: x.mode()[0]), E=("E", lambda x: x.mode()[0]),
                               C=("C", lambda x: x.mode()[0]), ASIL=("ASIL", lambda x: x.mode()[0]), n=("S", "size")).reset_index()
    cl = _client()
    try: cl.delete_collection("approved_hazards")
    except Exception: pass
    emb = Embedder()
    col = cl.create_collection("approved_hazards", embedding_function=emb)
    col.add(ids=[f"LIB-{i:04d}" for i in range(len(g))], documents=g.text.tolist(),
            metadatas=[{"S": int(r.S), "E": int(r.E), "C": int(r.C), "ASIL": r.ASIL, "n_reviews": int(r.n)} for r in g.itertuples()])
    return len(g), emb.name_used

_col = None
def similar(text, k=3):
    global _col
    if _col is None:
        _col = _client().get_collection("approved_hazards", embedding_function=Embedder())
    r = _col.query(query_texts=[text], n_results=k)
    return [{"id": i, "text": d, **m, "distance": round(float(x), 3)}
            for i, d, m, x in zip(r["ids"][0], r["documents"][0], r["metadatas"][0], r["distances"][0])]

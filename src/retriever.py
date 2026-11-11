"""Step 1 of GroundCheck: find the most relevant chunks of the document for a sentence,
using embeddings + cosine similarity."""
import numpy as np

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def pick_device():
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Retriever:
    def __init__(self, model_name="BAAI/bge-small-en-v1.5", device=None):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name, device=device or pick_device())

    def embed_passages(self, texts):
        return self.model.encode(texts, batch_size=64, normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=False)

    def embed_queries(self, texts):
        return self.embed_passages([QUERY_PREFIX + t for t in texts])

    @staticmethod
    def top_k(query_embs, chunk_embs, k=2):
        """Vectors are normalized, so dot product == cosine similarity.
        Returns, for each query, the indices of the k best chunks in document order."""
        sims = query_embs @ chunk_embs.T
        k = min(k, chunk_embs.shape[0])
        best = np.argsort(-sims, axis=1)[:, :k]
        return [sorted(row.tolist()) for row in best], sims

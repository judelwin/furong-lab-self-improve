from sentence_transformers import CrossEncoder
from config import RERANK_MODEL

class Reranker:
    def __init__(self):
        self.model = CrossEncoder(RERANK_MODEL)

    def rerank(self, query: str, hits):
        pairs = [(query, h["question"]) for h in hits]
        scores = self.model.predict(pairs)  # higher is better
        ranked = sorted(
            [{"rerank_score": float(s), **h} for s, h in zip(scores, hits)],
            key=lambda x: -x["rerank_score"]
        )
        return ranked

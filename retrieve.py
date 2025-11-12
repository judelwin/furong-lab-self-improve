import ujson, faiss
from sentence_transformers import SentenceTransformer
from config import FAISS_INDEX, IDS_JSON, JSONL_OUT, EMB_MODEL

def load_items_by_id():
    by_id = {}
    with open(JSONL_OUT, "r", encoding="utf-8") as f:
        for line in f:
            it = ujson.loads(line)
            by_id[it["id"]] = it
    return by_id

class Retriever:
    def __init__(self):
        self.model = SentenceTransformer(EMB_MODEL, device="cpu")
        self.index = faiss.read_index(FAISS_INDEX)
        with open(IDS_JSON, "r", encoding="utf-8") as f:
            self.ids = ujson.load(f)["ids"]
        self.items = load_items_by_id()

    def top_k(self, query: str, k: int = 10):
        qv = self.model.encode([query], normalize_embeddings=True)
        D, I = self.index.search(qv, k)
        hits = []
        for score, idx in zip(D[0], I[0]):
            if idx == -1: 
                continue
            doc_id = self.ids[idx]
            item = self.items[doc_id]
            hits.append({
                "id": doc_id,
                "score": float(score),
                "question": item["question"],
                "final_answer": item.get("final_answer"),
                "topic": item.get("topic"),
                "difficulty": item.get("difficulty"),
                "r1_short": (item.get("r1_solutions") or [""])[0]
            })
        return hits

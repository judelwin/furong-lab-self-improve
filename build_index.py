import ujson, faiss, numpy as np
import faulthandler; faulthandler.enable()
from sentence_transformers import SentenceTransformer
from config import JSONL_OUT, FAISS_INDEX, IDS_JSON, EMB_MODEL

def doc_text(item):
    # Text used for dense retrieval
    parts = [
        f"Question: {item['question']}",
        f"Topic: {item.get('topic','')}",
        f"Difficulty: {item.get('difficulty','')}",
        item.get("r1", ""),
        item.get("r2", ""),
        item.get("r3", "")
    ]
    return "\n".join([p for p in parts if p])

def main():
    print("Loading model...")
    model = SentenceTransformer(EMB_MODEL, device="cpu")
    print("Model loaded. Reading and encoding data in batches...")

    batch_size = 1000
    ids = []
    all_dim = None
    index = None
    batch_texts = []
    batch_ids = []
    total = 0
    with open(JSONL_OUT, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            it = ujson.loads(line)
            text = doc_text(it)
            if not isinstance(text, str) or text.strip() == "":
                print(f"[SKIP] Bad doc_text for id {it.get('id')}")
                continue
            batch_texts.append(text)
            batch_ids.append(it["id"])
            if len(batch_texts) == batch_size:
                print(f"Encoding batch {total//batch_size + 1} ({total + batch_size} total)...")
                X = model.encode(batch_texts, batch_size=len(batch_texts), normalize_embeddings=True)
                X = np.asarray(X, dtype="float32")
                if index is None:
                    all_dim = X.shape[1]
                    index = faiss.IndexFlatIP(all_dim)
                index.add(X)
                ids.extend(batch_ids)
                batch_texts, batch_ids = [], []
                total += batch_size
        # process any remaining items
        if batch_texts:
            print(f"Encoding final batch ({total + len(batch_texts)} total)...")
            X = model.encode(batch_texts, batch_size=len(batch_texts), normalize_embeddings=True)
            X = np.asarray(X, dtype="float32")
            if index is None:
                all_dim = X.shape[1]
                index = faiss.IndexFlatIP(all_dim)
            index.add(X)
            ids.extend(batch_ids)
            total += len(batch_texts)

    print("Writing FAISS index and ids...")
    faiss.write_index(index, FAISS_INDEX)
    with open(IDS_JSON, "w", encoding="utf-8") as f:
        ujson.dump({"ids": ids}, f)
    print(f"Indexed {len(ids)} items → {FAISS_INDEX}, ids → {IDS_JSON}")

if __name__ == "__main__":
    main()

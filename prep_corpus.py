import re, ujson, hashlib
from datasets import load_dataset
from config import HF_DATASET, JSONL_OUT

def clean(s):
    if s is None: return ""
    return re.sub(r"\s+\n", "\n", str(s).strip())

def make_id(row_text: str) -> str:
    # deterministic id for stability across runs
    h = hashlib.sha1(row_text.encode("utf-8")).hexdigest()[:16]
    return f"dm-{h}"

def main():
    ds = load_dataset(HF_DATASET, split="train")  # requires HF login if gated
    with open(JSONL_OUT, "w", encoding="utf-8") as f:
        for row in ds:
            question = clean(row.get("question"))
            final_answer = clean(row.get("final_answer"))
            topic = clean(row.get("topic"))
            difficulty = float(row.get("difficulty")) if row.get("difficulty") is not None else None

            r1s = [
                clean(row.get("r1_solution_1")),
                clean(row.get("r1_solution_2")),
                clean(row.get("r1_solution_3")),
            ]
            # build a stable id from core fields
            rid = make_id(question + final_answer + topic + str(difficulty))

            obj = {
                "id": rid,
                "question": question,
                "final_answer": final_answer,
                "difficulty": difficulty,
                "topic": topic,
                "r1_solutions": r1s,
            }
            f.write(ujson.dumps(obj, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()

import ujson

def print_and_save_topk(query, topk, output_file="topk_results.jsonl"):
    print(f"Query: {query}\n")
    print("Top-k similar questions:")
    # Use Google Drive path if running in Colab
    import os
    colab_drive_path = "/content/drive/MyDrive/DeepMathBackup/topk_results.jsonl"
    out_path = colab_drive_path if os.path.exists("/content/drive/MyDrive") else output_file
    with open(out_path, "w", encoding="utf-8") as f:
        for i, h in enumerate(topk):
            print(f"{i+1}. [id: {h['id']}] sim={h.get('score', '?'):.3f} topic={h.get('topic', '?')} diff={h.get('difficulty', '?')}")
            print(f"   Question: {h.get('question', '')}")
            out_obj = {
                "rank": i+1,
                "id": h["id"],
                "score": h.get("score", None),
                "topic": h.get("topic", None),
                "difficulty": h.get("difficulty", None),
                "question": h.get("question", None),
                "final_answer": h.get("final_answer", None)
            }
            f.write(ujson.dumps(out_obj, ensure_ascii=False) + "\n")
    print(f"Top-k results written to: {out_path}")

def build_references_block(chosen):
    blocks = []
    for h in chosen:
        r1 = (h.get("r1") or "").strip().replace("\n", " ")
        r2 = (h.get("r2") or "").strip().replace("\n", " ")
        r3 = (h.get("r3") or "").strip().replace("\n", " ")
        blocks.append(
            f"[{h['id']} | topic: {h.get('topic','?')} | diff: {h.get('difficulty','?')}]\n"
            f"Reference r1: {r1}\n"
            f"Reference r2: {r2}\n"
            f"Reference r3: {r3}\n"
            f"Final answer in ref: {h.get('final_answer','?')}\n"
        )
    return "\n".join(blocks)

def build_synthesis_prompt(user_q: str, references_block: str) -> str:
    return f"""Solve the problem step-by-step. If algebraic, show key transformations.
If numeric, check arithmetic. Return the final line as `Final Answer: ...` only once.

Problem:
{user_q}

Helpful references (adapt techniques, do not copy text):
{references_block}

When relevant, cite [ref: <id>] next to steps they inspired.
Finally, double-check and end with `Final Answer: ...`.
"""

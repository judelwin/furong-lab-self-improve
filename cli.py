import os, argparse
from config import K, N_CHOSEN
from retrieve import Retriever
from rerank import Reranker
from synthesize import print_and_save_topk

def run(query: str, k: int = K, n: int = N_CHOSEN):
    retr = Retriever()
    hits = retr.top_k(query, k=k)
    if not hits:
        print("No hits found for query.")
        return

    rr = Reranker()
    ranked = rr.rerank(query, hits)
    chosen = ranked[:n]

    # Print and save top-k results (with all relevant info)
    print_and_save_topk(query, chosen, output_file="topk_results.jsonl")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True, help="Math question to solve")
    ap.add_argument("--k", type=int, default=K)
    ap.add_argument("--n", type=int, default=N_CHOSEN)
    args = ap.parse_args()
    run(args.query, args.k, args.n)

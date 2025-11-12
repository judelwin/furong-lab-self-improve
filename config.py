EMB_MODEL =  "all-MiniLM-L6-v2" #"BAAI/bge-m3"           # strong general embedding (handles math text well)
RERANK_MODEL = "BAAI/bge-reranker-base"
HF_DATASET = "zwhe99/DeepMath-103K"

JSONL_OUT = "deepmath.jsonl"
FAISS_INDEX = "deepmath.faiss"
IDS_JSON = "ids.json"
K = 10           # initial retrieve
N_CHOSEN = 10     # after rerank

from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
texts = ["Test sentence 1", "Test sentence 2"]
embeddings = model.encode(texts, batch_size=2, normalize_embeddings=True)
print(embeddings.shape)